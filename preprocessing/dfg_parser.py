class DfgParser:

	def __init__(self):
		self.assignment = ['assignment', 'augmented_assignment', 'for_in_clause']
		self.if_statement = ['if_statement']
		self.for_statement = ['for_statement']
		self.while_statement = ['while_statement']
		self.do_first_statement = ['for_in_clause']
		self.def_statement = ['default_parameter']

	def tree_to_variable_index(self, root_node, ast_tok_index_to_code_tok):
		if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
			index = (root_node.start_point, root_node.end_point)
			_, code = ast_tok_index_to_code_tok[index]
			if root_node.type != code:
				return [index]
			else:
				return []
		else:
			code_tokens = []
			for child in root_node.children:
				code_tokens += self.tree_to_variable_index(child, ast_tok_index_to_code_tok)
			return code_tokens

	def parse_leave(self, root_node, ast_tok_index_to_code_tok, states):
		idx, code_tok = ast_tok_index_to_code_tok[(root_node.start_point, root_node.end_point)]
		if root_node.type == code_tok:
			return [], states
		elif code_tok in states:
			return [(code_tok, idx, 'comesFrom', [code_tok], states[code_tok].copy())], states
		else:
			if root_node.type == 'identifier':
				states[code_tok] = [idx]
			return [(code_tok, idx, 'comesFrom', [], [])], states

	def parse_def_statement(self, root_node, ast_tok_index_to_code_tok, states):
		name = root_node.child_by_field_name('name')
		value = root_node.child_by_field_name('value')
		dfg = []

		if value is None:
			indexs = self.tree_to_variable_index(name, ast_tok_index_to_code_tok)
			for index in indexs:
				idx, code_tok = ast_tok_index_to_code_tok[index]
				dfg.append((code_tok, idx, 'comesFrom', [], []))
				states[code_tok] = [idx]
			return sorted(dfg, key=lambda x: x[1]), states

		else:
			name_indexs = self.tree_to_variable_index(name, ast_tok_index_to_code_tok)
			value_indexs = self.tree_to_variable_index(value, ast_tok_index_to_code_tok)
			temp, states = self.parse_dfg_python(value, ast_tok_index_to_code_tok, states)
			dfg += temp
			for index1 in name_indexs:
				idx1, code1 = ast_tok_index_to_code_tok[index1]
				for index2 in value_indexs:
					idx2, code2 = ast_tok_index_to_code_tok[index2]
					dfg.append((code1, idx1, 'comesFrom', [code2], [idx2]))
				states[code1] = [idx1]
			return sorted(dfg, key=lambda x: x[1]), states

	def parse_assignment(self, root_node, ast_tok_index_to_code_tok, states):
		if root_node.type == 'for_in_clause':
			right_nodes = [root_node.children[-1]]
			left_nodes = [root_node.child_by_field_name('left')]
		else:
			if root_node.child_by_field_name('right') is None:
				return [], states
			left_nodes = [x for x in root_node.child_by_field_name('left').children if x.type != ',']
			right_nodes = [x for x in root_node.child_by_field_name('right').children if x.type != ',']
			if len(right_nodes) != len(left_nodes) or not any(node.type == "identifier" for node in right_nodes):
				left_nodes = [root_node.child_by_field_name('left')]
				right_nodes = [root_node.child_by_field_name('right')]
			if len(left_nodes) == 0:
				left_nodes = [root_node.child_by_field_name('left')]
			if len(right_nodes) == 0:
				right_nodes = [root_node.child_by_field_name('right')]

		dfg = []
		for node in right_nodes:
			temp, states = self.parse_dfg_python(node, ast_tok_index_to_code_tok, states)
			dfg += temp

		for left_node, right_node in zip(left_nodes, right_nodes):
			left_tokens_index = self.tree_to_variable_index(left_node, ast_tok_index_to_code_tok)
			right_tokens_index = self.tree_to_variable_index(right_node, ast_tok_index_to_code_tok)
			temp = []
			for token1_index in left_tokens_index:
				idx1, code1 = ast_tok_index_to_code_tok[token1_index]
				temp.append((code1, idx1, 'computedFrom', [ast_tok_index_to_code_tok[x][1] for x in right_tokens_index],
							 [ast_tok_index_to_code_tok[x][0] for x in right_tokens_index]))
				states[code1] = [idx1]
			dfg += temp

		return sorted(dfg, key=lambda x: x[1]), states

	def parse_if_statement(self, states, root_node, ast_tok_index_to_code_tok):
		dfg = []
		current_states = states.copy()
		others_states = []
		tag = False

		if 'else' in root_node.type:
			tag = True
		for child in root_node.children:
			if 'else' in child.type:
				tag = True
			if child.type not in ['elif_clause', 'else_clause']:
				temp, current_states = self.parse_dfg_python(child, ast_tok_index_to_code_tok, current_states)
				dfg += temp
			else:
				temp, new_states = self.parse_dfg_python(child, ast_tok_index_to_code_tok, states)
				dfg += temp
				others_states.append(new_states)

		others_states.append(current_states)
		if tag is False:
			others_states.append(states)
		new_states = {}

		for dic in others_states:
			for key in dic:
				if key not in new_states:
					new_states[key] = dic[key].copy()
				else:
					new_states[key] += dic[key]

		for key in new_states:
			new_states[key] = sorted(list(set(new_states[key])))

		return sorted(dfg, key=lambda x: x[1]), new_states

	def parse_for_statement(self, root_node, ast_tok_index_to_code_tok, states):
		dfg = []
		for i in range(2):
			right_nodes = [x for x in root_node.child_by_field_name('right').children if x.type != ',']
			left_nodes = [x for x in root_node.child_by_field_name('left').children if x.type != ',']
			if len(right_nodes) != len(left_nodes) or not any(node.type == "identifier" for node in right_nodes):
				left_nodes = [root_node.child_by_field_name('left')]
				right_nodes = [root_node.child_by_field_name('right')]
			if len(left_nodes) == 0:
				left_nodes = [root_node.child_by_field_name('left')]
			if len(right_nodes) == 0:
				right_nodes = [root_node.child_by_field_name('right')]

			for node in right_nodes:
				temp, states = self.parse_dfg_python(node, ast_tok_index_to_code_tok, states)
				dfg += temp

			for left_node, right_node in zip(left_nodes, right_nodes):
				left_tokens_index = self.tree_to_variable_index(left_node, ast_tok_index_to_code_tok)
				right_tokens_index = self.tree_to_variable_index(right_node, ast_tok_index_to_code_tok)
				temp = []

				for token1_index in left_tokens_index:
					idx1, code1 = ast_tok_index_to_code_tok[token1_index]
					temp.append(
						(code1, idx1, 'computedFrom', [ast_tok_index_to_code_tok[x][1] for x in right_tokens_index],
						 [ast_tok_index_to_code_tok[x][0] for x in right_tokens_index]))
					states[code1] = [idx1]

				dfg += temp

			if root_node.children[-1].type == "block":
				temp, states = self.parse_dfg_python(root_node.children[-1], ast_tok_index_to_code_tok, states)
				dfg += temp

		dic = {}
		for x in dfg:
			if (x[0], x[1], x[2]) not in dic:
				dic[(x[0], x[1], x[2])] = [x[3], x[4]]
			else:
				dic[(x[0], x[1], x[2])][0] = list(set(dic[(x[0], x[1], x[2])][0] + x[3]))
				dic[(x[0], x[1], x[2])][1] = sorted(list(set(dic[(x[0], x[1], x[2])][1] + x[4])))

		dfg = [(x[0], x[1], x[2], y[0], y[1]) for x, y in sorted(dic.items(), key=lambda t: t[0][1])]

		return sorted(dfg, key=lambda x: x[1]), states

	def parse_while_statement(self, root_node, ast_tok_index_to_code_tok, states):
		dfg = []
		for i in range(2):
			for child in root_node.children:
				temp, states = self.parse_dfg_python(child, ast_tok_index_to_code_tok, states)
				dfg += temp

		dic = {}
		for x in dfg:
			if (x[0], x[1], x[2]) not in dic:
				dic[(x[0], x[1], x[2])] = [x[3], x[4]]
			else:
				dic[(x[0], x[1], x[2])][0] = list(set(dic[(x[0], x[1], x[2])][0] + x[3]))
				dic[(x[0], x[1], x[2])][1] = sorted(list(set(dic[(x[0], x[1], x[2])][1] + x[4])))

		dfg = [(x[0], x[1], x[2], y[0], y[1]) for x, y in sorted(dic.items(), key=lambda t: t[0][1])]

		return sorted(dfg, key=lambda x: x[1]), states

	def parse_other(self, root_node, ast_tok_index_to_code_tok, states):
		dfg = []
		for child in root_node.children:
			if child.type in self.do_first_statement:
				temp, states = self.parse_dfg_python(child, ast_tok_index_to_code_tok, states)
				dfg += temp
		for child in root_node.children:
			if child.type not in self.do_first_statement:
				temp, states = self.parse_dfg_python(child, ast_tok_index_to_code_tok, states)
				dfg += temp

		return sorted(dfg, key=lambda x: x[1]), states

	def parse_dfg_python(self, root_node, ast_tok_index_to_code_tok, states):
		states = states.copy()

		if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
			return self.parse_leave(root_node, ast_tok_index_to_code_tok, states)
		elif root_node.type in self.def_statement:
			return self.parse_def_statement(root_node, ast_tok_index_to_code_tok, states)
		elif root_node.type in self.assignment:
			return self.parse_assignment(root_node, ast_tok_index_to_code_tok, states)
		elif root_node.type in self.if_statement:
			return self.parse_if_statement(states, root_node, ast_tok_index_to_code_tok)
		elif root_node.type in self.for_statement:
			return self.parse_for_statement(root_node, ast_tok_index_to_code_tok, states)
		elif root_node.type in self.while_statement:
			return self.parse_while_statement(root_node, ast_tok_index_to_code_tok, states)
		else:
			return self.parse_other(root_node, ast_tok_index_to_code_tok, states)