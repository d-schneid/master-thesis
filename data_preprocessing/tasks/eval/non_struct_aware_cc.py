from data_preprocessing.tasks.code_completion import CodeCompletion


class NonStructAwareCC(CodeCompletion):

	def _reset_floats(self, batch_no_labels, batch):
		batch_no_labels["attention_bias"][batch_no_labels["attention_bias"] > -1] = self.attn_bias_attend
		batch_no_labels["attention_bias"][batch_no_labels["attention_bias"] <= -1] = self.attn_bias_ignore
