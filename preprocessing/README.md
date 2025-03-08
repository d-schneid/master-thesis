### Data Structures After Preprocessing

**code_tokens**: Tokenized code based on the used tokenizer. These tokens are used to compute the self-attention between code tokens.

**text_tokens**: Tokenized text (corresponding to the code) based on the used tokenizer.

**ast_leaf_code_token_idxs**: 1:N mapping of the respective AST leaf to the indices of the corresponding code tokens in **code_tokens**. Based on this information, the token-leaf linking matrix can be derived. This matrix is needed to compute the self-attention between code tokens and AST leaves.

**ll_sims**: Similarity score (w/o log(1 + ...) of each AST leaf to each other AST leaf based on the number of common nodes on the respective leaf-root paths. The information is pruned to only contain the upper triangular matrix w/o diagonals, since this similarity score is symmetric and diagonal entries are always 1. This similarity score is needed to compute the self-attention between two AST leaves.

**lr_paths_types**: For each leaf-root path, it contains a list containing the custom indices of the corresponding types of nodes on the respective leaf-root path. These node types are needed to embed the corresponding AST leaf.

**dfg_node_code_token_idxs**: 1:N mapping of the respective DFG node to the indices of the corresponding code tokens in **code_tokens**. Based on this information, the token-variable linking matrix can be derived. This matrix is needed to compute the self-attention between code tokens and DFG nodes.

**dfg_edges**: Contains a tuple for each edge in the corresponding DFG. The tuples contain the custom indices of the corresponding DFG nodes. These custom indices can be used to index **dfg_node_code_token_idxs** to get the corresponding code token indices for each DFG node. Semantics of a tuple: The left tuple entry corresponds to a DFG node that comes from the right tuple entry, which can correspond to multiple DFG nodes. Based on the **dfg_edges**, the DFG adjacency matrix can be derived. This matrix is needed to compute the self-attention between DFG nodes.
