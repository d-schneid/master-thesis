import ast

from data_preprocessing.data_handler import DataHandler


if __name__ == '__main__':
	data_dir = '../data/pretraining/'
	data_handler = DataHandler(save_dir=data_dir)
	data = data_handler.get_concat_stored_data()

	row = 93

	code_tokens = list(map(int, data.at[row, 'code_tokens'].split(',')))
	lr_paths_len = list(map(int, data.at[row, 'lr_paths_len'].split(',')))
	dfg_node_mask = list(map(int, data.at[row, 'dfg_node_mask'].split(',')))
	num_total_tokens = len(code_tokens) + len(lr_paths_len) + len(dfg_node_mask)

	# get individual attention masks
	attn_code_tokens = ast.literal_eval(data.at[row, 'attn_code_tokens'])
	attn_ast_leaves = ast.literal_eval(data.at[row, 'attn_ast_leaves'])
	attn_dfg_edges = ast.literal_eval(data.at[row, 'attn_dfg_edges'])
	attn_code_ast = ast.literal_eval(data.at[row, 'attn_code_ast'])
	attn_code_dfg = ast.literal_eval(data.at[row, 'attn_code_dfg'])

	# compute transpose
	attn_code_ast_T = [list(row) for row in zip(*attn_code_ast)]
	attn_code_dfg_T = [list(row) for row in zip(*attn_code_dfg)]

	# compute null matrix for attention between AST leaves and DFG edges
	num_rows = len(attn_ast_leaves)
	num_cols = len(attn_dfg_edges[0])
	attn_ast_dfg = [[0] * num_cols for _ in range(num_rows)]
	attn_ast_dfg_T = [list(row) for row in zip(*attn_ast_dfg)]

	# build block matrices column-wise
	first_col_matrix = attn_code_tokens + attn_code_ast_T + attn_code_dfg_T
	second_col_matrix = attn_code_ast + attn_ast_leaves + attn_ast_dfg_T
	third_col_matrix = attn_code_dfg + attn_ast_dfg + attn_dfg_edges

	# symmetric matrix
	attn_mask = [row1 + row2 + row3 for row1, row2, row3 in zip(first_col_matrix, second_col_matrix, third_col_matrix)]

	print("hello")
