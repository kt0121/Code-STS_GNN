import torch


class AttentionModule(torch.nn.Module):
    def __init__(self, in_channel):

        super(AttentionModule, self).__init__()
        self.in_channel = in_channel
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):

        self.weight_matrix = torch.nn.Parameter(
            torch.Tensor(self.in_channel, self.in_channel)
        )

    def init_parameters(self):

        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):

        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=0)
        transformed_global = torch.tanh(global_context)
        sigmoid_scores = torch.sigmoid(
            torch.mm(embedding, transformed_global.view(-1, 1))
        )
        representation = torch.mm(torch.t(embedding), sigmoid_scores)
        return representation


class TenorNetworkModule(torch.nn.Module):
    def __init__(self, in_channel):

        super(TenorNetworkModule, self).__init__()
        self.in_channel = in_channel
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):

        self.weight_matrix = torch.nn.Parameter(
            torch.Tensor(self.in_channel, self.in_channel, 16)
        )

        self.weight_matrix_block = torch.nn.Parameter(
            torch.Tensor(16, 2 * self.in_channel)
        )
        self.bias = torch.nn.Parameter(torch.Tensor(16, 1))

    def init_parameters(self):

        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):

        scoring = torch.mm(
            torch.t(embedding_1), self.weight_matrix.view(self.in_channel, -1)
        )
        scoring = scoring.view(self.in_channel, 16)
        scoring = torch.mm(torch.t(scoring), embedding_2)
        combined_representation = torch.cat((embedding_1, embedding_2))
        block_scoring = torch.mm(self.weight_matrix_block, combined_representation)
        scores = torch.nn.functional.tanh(scoring + block_scoring + self.bias)
        return scores
