class DfgParser:

	def __init__(self):
		self.assignment = ['assignment', 'augmented_assignment', 'for_in_clause']
		self.if_statement = ['if_statement']
		self.for_statement = ['for_statement']
		self.while_statement = ['while_statement']
		self.do_first_statement = ['for_in_clause']
		self.def_statement = ['default_parameter']
		self.keyword_argument = ['keyword_argument']
		self.lambda_ = ['lambda']
		self.comprehension = ['dictionary_comprehension', 'list_comprehension', 'lambda']

	def reduce_dfg_edges(self, dfg):
		dic = {}
		for x in dfg:
			if (x[0], x[1], x[2]) not in dic:
				dic[(x[0], x[1], x[2])] = [x[3], x[4]]
			else:
				dic[(x[0], x[1], x[2])][0], dic[(x[0], x[1], x[2])][1] = self.merge_lists_stably(
					dic[(x[0], x[1], x[2])][0], dic[(x[0], x[1], x[2])][1], x[3], x[4])

		dfg = [(x[0], x[1], x[2], y[0], y[1]) for x, y in sorted(dic.items(), key=lambda t: t[0][1])]

		return dfg

	def merge_lists_stably(self, existing_strings, existing_integers, new_strings, new_integers):
		seen_strings = set(existing_strings)
		merged_strings = existing_strings.copy()
		merged_integers = existing_integers.copy()

		for new_str, new_int in zip(new_strings, new_integers):
			if new_str not in seen_strings:
				merged_strings.append(new_str)
				merged_integers.append(new_int)
				seen_strings.add(new_str)

		sorted_pairs = sorted(zip(merged_integers, merged_strings))
		if not sorted_pairs: return [], []
		sorted_integers, sorted_strings = zip(*sorted_pairs)

		return list(sorted_strings), list(sorted_integers)

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

	def parse_leave(self, root_node, ast_tok_index_to_code_tok, states, no_new_states):
		idx, code_tok = ast_tok_index_to_code_tok[(root_node.start_point, root_node.end_point)]
		if root_node.type.lower() == code_tok.lower():
			return [], states
		# account for naming collision: when there is a '.' before a variable it only belongs to the respective object
		# calling the variable and not to a variable declared in the code
		elif code_tok in states and (root_node.prev_sibling is None or root_node.prev_sibling.type != '.'):
			return [(code_tok, idx, 'comesFrom', [code_tok], states[code_tok].copy())], states
		elif root_node.type == 'identifier' and no_new_states and root_node.parent.type != 'lambda_parameters':
			if root_node.type == 'identifier': return [(code_tok, idx, 'comesFrom', [], [])], states
		else:
			if root_node.type == 'identifier': states[code_tok] = [idx]
			return [(code_tok, idx, 'comesFrom', [], [])], states

	def parse_def_statement(self, root_node, ast_tok_index_to_code_tok, states, no_new_states):
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
			temp, states = self.parse_dfg_python(value, ast_tok_index_to_code_tok, states, no_new_states)
			dfg += temp
			for index1 in name_indexs:
				idx1, code1 = ast_tok_index_to_code_tok[index1]
				for index2 in value_indexs:
					idx2, code2 = ast_tok_index_to_code_tok[index2]
					dfg.append((code1, idx1, 'comesFrom', [code2], [idx2]))
				states[code1] = [idx1]
			return sorted(dfg, key=lambda x: x[1]), states

	def parse_assignment(self, root_node, ast_tok_index_to_code_tok, states, no_new_states):
		if root_node.type == 'for_in_clause':
			right_nodes = [root_node.children[-1]]
			left_nodes = [root_node.child_by_field_name('left')]
		elif root_node.type == 'keyword_argument':
			left_nodes = [root_node.children[0]]
			right_nodes = [root_node.children[-1]]
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
			temp, states = self.parse_dfg_python(node, ast_tok_index_to_code_tok, states, no_new_states=True)
			dfg += temp

		for left_node, right_node in zip(left_nodes, right_nodes):
			left_tokens_index = self.tree_to_variable_index(left_node, ast_tok_index_to_code_tok)
			right_tokens_index = self.tree_to_variable_index(right_node, ast_tok_index_to_code_tok)
			temp = []
			for token1_index in left_tokens_index:
				idx1, code1 = ast_tok_index_to_code_tok[token1_index]
				temp.append((code1, idx1, 'computedFrom', [ast_tok_index_to_code_tok[x][1] for x in right_tokens_index],
							 [ast_tok_index_to_code_tok[x][0] for x in right_tokens_index]))
				if not no_new_states: states[code1] = [idx1]
			dfg += temp

		return sorted(dfg, key=lambda x: x[1]), states

	def parse_if_statement(self, states, root_node, ast_tok_index_to_code_tok, no_new_states):
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
				temp, current_states = self.parse_dfg_python(child, ast_tok_index_to_code_tok, current_states, no_new_states)
				dfg += temp
			else:
				temp, new_states = self.parse_dfg_python(child, ast_tok_index_to_code_tok, states, no_new_states)
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

	def parse_for_statement(self, root_node, ast_tok_index_to_code_tok, states, no_new_states):
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
				temp, states = self.parse_dfg_python(node, ast_tok_index_to_code_tok, states, no_new_states)
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
				temp, states = self.parse_dfg_python(root_node.children[-1], ast_tok_index_to_code_tok, states, no_new_states)
				dfg += temp

		dfg = self.reduce_dfg_edges(dfg)

		return sorted(dfg, key=lambda x: x[1]), states

	def parse_while_statement(self, root_node, ast_tok_index_to_code_tok, states, no_new_states):
		dfg = []
		for i in range(2):
			for child in root_node.children:
				temp, states = self.parse_dfg_python(child, ast_tok_index_to_code_tok, states, no_new_states)
				dfg += temp

		dfg = self.reduce_dfg_edges(dfg)

		return sorted(dfg, key=lambda x: x[1]), states

	def parse_lambda(self, root_node, ast_tok_index_to_code_tok, states, no_new_states):
		dfg = []
		current_states = states.copy()
		for child in root_node.children:
			temp, current_states = self.parse_dfg_python(child, ast_tok_index_to_code_tok, current_states, no_new_states)
			dfg += temp

		return sorted(dfg, key=lambda x: x[1]), states

	def parse_comprehension(self, root_node, ast_tok_index_to_code_tok, states, no_new_states):
		dfg = []
		current_states = states.copy()

		for child in root_node.children:
			if child.type in self.do_first_statement:
				temp, current_states = self.parse_dfg_python(child, ast_tok_index_to_code_tok, current_states, no_new_states=False)
				dfg += temp

		for child in root_node.children:
			if child.type not in self.do_first_statement:
				curr_no_new_states = True if child.type == 'pair' else False
				temp, current_states = self.parse_dfg_python(child, ast_tok_index_to_code_tok, current_states, no_new_states=curr_no_new_states)
				dfg += temp

		return sorted(dfg, key=lambda x: x[1]), states

	def parse_other(self, root_node, ast_tok_index_to_code_tok, states, no_new_states):
		if root_node.type == 'call': no_new_states = True # a call can never declare new variables
		dfg = []

		for child in root_node.children:
			if child.type in self.do_first_statement:
				temp, states = self.parse_dfg_python(child, ast_tok_index_to_code_tok, states, no_new_states)
				dfg += temp

		for child in root_node.children:
			if child.type not in self.do_first_statement:
				temp, states = self.parse_dfg_python(child, ast_tok_index_to_code_tok, states, no_new_states)
				dfg += temp

		return sorted(dfg, key=lambda x: x[1]), states

	def parse_dfg_python(self, root_node, ast_tok_index_to_code_tok, states, no_new_states=False):
		states = states.copy()

		if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
			return self.parse_leave(root_node, ast_tok_index_to_code_tok, states, no_new_states)
		elif root_node.type in self.def_statement:
			return self.parse_def_statement(root_node, ast_tok_index_to_code_tok, states, no_new_states)
		elif root_node.type in self.assignment or root_node.type in self.keyword_argument:
			return self.parse_assignment(root_node, ast_tok_index_to_code_tok, states, no_new_states)
		elif root_node.type in self.if_statement:
			return self.parse_if_statement(states, root_node, ast_tok_index_to_code_tok, no_new_states)
		elif root_node.type in self.for_statement:
			return self.parse_for_statement(root_node, ast_tok_index_to_code_tok, states, no_new_states)
		elif root_node.type in self.while_statement:
			return self.parse_while_statement(root_node, ast_tok_index_to_code_tok, states, no_new_states)
		#elif root_node.type in self.lambda_:
		#	return self.parse_lambda(root_node, ast_tok_index_to_code_tok, states, no_new_states)
		elif root_node.type in self.comprehension:
			return self.parse_comprehension(root_node, ast_tok_index_to_code_tok, states, no_new_states)
		else:
			return self.parse_other(root_node, ast_tok_index_to_code_tok, states, no_new_states)
