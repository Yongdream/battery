import torch
import torch.nn as nn
import torch.nn.functional as F


class ADDAModel(nn.Module):
    def __init__(self, input_shape=(300, 12), input_shape_domain=(300, 12), pretrain=True):
        super(ADDAModel, self).__init__()

        self.Conv1D_1 = nn.Conv1d(12, 64, kernel_size=64, stride=16, padding=16)
        self.ReLU_1 = nn.ReLU()
        self.MaxPooling1D_1 = nn.MaxPool1d(kernel_size=2)

        self.Conv1D_2 = nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1)
        self.BN_2 = nn.BatchNorm1d(32)
        self.ReLU_2 = nn.ReLU()
        self.MaxPooling1D_2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.Flatten_1 = nn.Flatten()

        self.Conv1D_1_domain = nn.Conv1d(12, 64, kernel_size=64, stride=16, padding=16)
        self.ReLU_1_domain = nn.ReLU()
        self.MaxPooling1D_1_domain = nn.MaxPool1d(kernel_size=2)

        self.Conv1D_2_domain = nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1)
        self.BN_2_domain = nn.BatchNorm1d(32)
        self.ReLU_2_domain = nn.ReLU()
        self.MaxPooling1D_2_domain = nn.MaxPool1d(kernel_size=2, stride=2)

        self.Flatten_1_domain = nn.Flatten()
        self.Concatenate_domain = nn.Concatenate(dim=0)

        self.Dense_1 = nn.Linear(64, 32)
        self.ReLU_3 = nn.ReLU()
        self.pred_loss = nn.Linear(32, 5)

        self.Dense_2 = nn.Linear(64, 32)
        self.ReLU_5 = nn.ReLU()
        self.Dense_3 = nn.Linear(32, 5)
        self.domain_loss = nn.Linear(5, 1)

        self.pretrain = pretrain

    def source_feature_extractor_model(self, input_tensor):
        Conv1D_1 = self.Conv1D_1(input_tensor)
        ReLU_1 = self.ReLU_1(Conv1D_1)
        MaxPooling1D_1 = self.MaxPooling1D_1(ReLU_1)

        Conv1D_2 = self.Conv1D_2(MaxPooling1D_1)
        BN_2 = self.BN_2(Conv1D_2)
        ReLU_2 = self.ReLU_2(BN_2)
        MaxPooling1D_2 = self.MaxPooling1D_2(ReLU_2)
        feature = self.Flatten_1(MaxPooling1D_2)
        return feature

    def domain_feature_extractor_model(self, input_tensor):
        Conv1D_1_domain = self.Conv1D_1_domain(input_tensor)
        ReLU_1_domain = self.ReLU_1_domain(Conv1D_1_domain)
        MaxPooling1D_1_domain = self.MaxPooling1D_1_domain(ReLU_1_domain)

        Conv1D_2_domain = self.Conv1D_2_domain(MaxPooling1D_1_domain)
        BN_2_domain = self.BN_2_domain(Conv1D_2_domain)
        MaxPooling1D_2_domain = self.MaxPooling1D_2_domain(BN_2_domain)
        feature = self.Flatten_1_domain(MaxPooling1D_2_domain)
        return feature

    def label_predictor_model(self, feature):
        Dense_1 = self.Dense_1(feature)
        ReLU_3 = self.ReLU_3(Dense_1)
        pred_loss = self.pred_loss(ReLU_3)
        return pred_loss

    def domain_predictor_model(self, feature):
        Dense_2 = self.Dense_2(feature)
        Dense_3 = self.Dense_3(Dense_2)
        ReLU_5 = self.ReLU_5(Dense_3)
        domain_loss = self.domain_loss(ReLU_5)
        return domain_loss

    def forward(self, input_tensor):
        source_feature = self.source_feature_extractor_model(input_tensor[0])
        domain_feature = self.domain_feature_extractor_model(input_tensor[1])
        feature = self.Concatenate_domain([source_feature, domain_feature])
        domain = self.domain_predictor_model(feature)
        return domain

    def recall(self, input_tensor):
        source_feature = self.source_feature_extractor_model(input_tensor[0])
        domain_feature = self.domain_feature_extractor_model(input_tensor[1])
        domain = self.domain_predictor_model(domain_feature)
        return domain

    def vaildmodel(self, input_tensor):
        domain_feature = self.domain_feature_extractor_model(input_tensor)
        pred = self.label_predictor_model(domain_feature)
        return pred

    def pretrain(self, input_tensor):
        source_feature = self.source_feature_extractor_model(input_tensor[0])
        self.pred = self.label_predictor_model(source_feature)
        return self.pred


model = ADDAModel(pretrain=True)
