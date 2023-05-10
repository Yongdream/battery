import torch
import torch.nn as nn


# 自定义模型
class CBDANModel(nn.Module):
    # 可以传入一些超参数，用以动态构建模型
    # __init_——()方法在创建模型对象时被调用diagnosis
    # input_shape: 输入层和输出层的节点个数（输入层实际要比这多1，因为有个bias）
    # hidden_shape: 隐藏层节点个数，隐藏层节点的最后一个节点值固定为1，也是bias
    # 使用方法：直接传入实际的input_shape即可，在call中也直接传入原始Input_tensor即可
    # 一切关于数据适配模型的处理都在模型中实现
    def __init__(self, input_shape=(300, 12)):
        # 调用父类__init__()方法
        super(CBDANModel, self).__init__()

        # 第一层卷积
        self.Conv1D_1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size =8, stride=1, padding='same')
        self.ReLU_1 = nn.ReLU()
        self.MaxPooling1D_1 = nn.MaxPool1d(kernel_size=2)
        self.Dropout_1 = nn.Dropout(0.5)

        # 第二层卷积
        self.Conv1D_2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.BN_2 = nn.BatchNorm1d(32)
        self.ReLU_2 = nn.ReLU()
        self.MaxPooling1D_2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.Dropout_2 = nn.Dropout(0.5)

        self.GlobalAveragePooling1D = nn.AdaptiveAvgPool1d(1)
        self.GlobalMaxPool1D = nn.AdaptiveMaxPool1d(1)

        # Shared MLP
        self.Dense_layer1 = nn.Linear(in_features=64, out_features=8)
        self.Dense_layer2 = nn.Linear(in_features=8, out_features=32)

        self.Concatenate = torch.cat
        self.spatial_conv1d = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=7, stride=1, padding='same')
        self.Flatten_1 = nn.Flatten()

        # 用于类预测的MLP
        self.Dense_1 = nn.Linear(in_features=32, out_features=32)
        self.ReLU_3 = nn.ReLU()

        self.pred_loss = nn.Sequential(
            nn.Linear(in_features=32, out_features=5),
            nn.Softmax(dim=1)
        )

        # 具有对抗性损失的域预测小MLP算法
        self.Concatenate_pred = torch.cat
        self.Dense_2 = nn.Linear(in_features=64, out_features=32)
        self.ReLU_5 = nn.ReLU()
        self.Dense_3 = nn.Linear(in_features=32, out_features=5)
        self.domain_loss = nn.Sequential(
            nn.Linear(in_features=5, out_features=1),
            nn.Sigmoid()
        )

        self.add = nn.quantized.FloatFunctional()
        self.sigmoid = nn.Sigmoid()

        self.avgpool_spatial = nn.AdaptiveAvgPool1d(output_size=1)
        self.maxpool_spatial = nn.AdaptiveMaxPool1d(output_size=1)

    def CBAM(self, input_tensor):
        # Channel Attention
        avgpool = self.GlobalAveragePooling1D(input_tensor)
        maxpool = self.GlobalMaxPool1D(input_tensor)
        # Shared MLP
        avg_out = self.Dense_layer2(self.Dense_layer1(avgpool))
        max_out = self.Dense_layer2(self.Dense_layer1(maxpool))

        channel1 = self.add([avg_out, max_out])
        channel2 = self.sigmoid(channel1)
        channel3 = channel2.view(-1, 1, 32)
        channel_out = torch.mul(input_tensor, channel3)

        # Spatial Attention
        avgpool_Spatial = self.avgpool_spatial(channel_out)
        maxpool_Spatial = self.maxpool_spatial(channel_out)
        spatial = self.Concatenate([avgpool_Spatial, maxpool_Spatial])

        spatial1 = self.spatial_conv1d(spatial)
        spatial_out = self.sigmoid(spatial1, name='spatial_sigmoid')

        CBAM_out = torch.mul(channel_out, spatial_out)
        return CBAM_out

    def feature_extractor_model(self, input_tensor):
        # 特征提取
        Conv1D_1 = self.Conv1D_1(input_tensor)
        ReLU_1 = self.ReLU_1(Conv1D_1)
        MaxPooling1D_1 = self.MaxPooling1D_1(ReLU_1)
        Conv1D_2 = self.Conv1D_2(MaxPooling1D_1)
        BN_2 = self.BN_2(Conv1D_2)
        ReLU_2 = self.ReLU_2(BN_2)
        MaxPooling1D_2 = self.MaxPooling1D_2(ReLU_2)
        CBAM_layer = self.CBAM(MaxPooling1D_2)
        feature = self.Flatten_1(CBAM_layer)

        return feature

    def label_predictor_model(self, feature, training=tf.cast(False, tf.bool)):
        # 切换到路由目标示例(second half of batch) differently
        # 取决于训练或测试模式。
        all_features = lambda: feature
        source_features = lambda: feature
        classify_feats = tf.cond(training, source_features, all_features)
        # 用于类预测的MLP
        Dense_1 = self.Dense_1(classify_feats)
        ReLU_3 = self.ReLU_3(Dense_1)
        pred_loss = self.pred_loss(ReLU_3)
        return pred_loss

    #     @tf.custom_gradient
    #     def FlipGradientBuilder(self,x,l=1):
    #         def grad(neg):
    #             print("_flip_gradients")
    #             return [tf.negative(neg) * l]
    #         y = tf.identity(x)
    #         print(y)
    #         return y,grad
    def FlipGradientBuilder(self, x, l=1):
        y = tf.identity(x)
        print(y)
        return y

    def domain_predictor_model(self, feature, pred, training=tf.cast(False, tf.bool)):
        # 反向梯度反转
        z = -feature
        feat = z + tf.stop_gradient(feature - z)
        # feat = self.FlipGradientBuilder(feature)
        # print(feat)
        # 具有对抗性损失的域预测小MLP算法
        Dense_2 = self.Dense_2(feat)
        Dense_3 = self.Dense_3(Dense_2)
        Concatenate_pred = layers.add([Dense_3, pred], name='addAll')
        # Concatenate_pred = self.Concatenate_pred([Dense_3,pred])
        ReLU_5 = self.ReLU_5(Concatenate_pred)
        domain_loss = self.domain_loss(ReLU_5)
        return domain_loss

    def call(self, input_tensor, training=False):
        # 特征提取
        self.feature = self.feature_extractor_model(input_tensor)
        # 用于类预测的MLP
        self.pred = self.label_predictor_model(self.feature)
        # 具有对抗性损失的域预测小MLP算法
        self.domain = self.domain_predictor_model(self.feature, self.pred)

        return self.pred, self.domain

    # 计算损失值
    def categorical_crossentropy(self, output, target, axis=-1):
        epsilon = backend_config.epsilon
        if (isinstance(output, (ops.EagerTensor, variables_module.Variable)) or
                output.op.type != 'Softmax'):
            # 缩放Pred，使每个样本的类概率总和为1
            output = output / math_ops.reduce_sum(output, axis, True)
            # 根据概率计算交叉熵。
            epsilon_ = constant_op.constant(epsilon(), output.dtype.base_dtype)
            output = clip_ops.clip_by_value(output, epsilon_, 1. - epsilon_)
            return -math_ops.reduce_sum(target * math_ops.log(output), axis)

    def get_loss(self, true_pre):
        index_pre = true_pre[0]
        type_pre = true_pre[1]

        # index_loss = self.categorical_crossentropy(tf.constant(self.pred, dtype=tf.float32),tf.constant(index_pre, dtype=tf.float32))
        # Softmax + categorical_crossentropy可以用于二分类等价于Sigmoid + binary_crossentropy二分类
        # type_loss = self.categorical_crossentropy(tf.constant(self.domain, dtype=tf.float32),tf.constant(type_pre, dtype=tf.float32))
        losses = [l(t, o) for l, o, t in
                  zip([tf.keras.losses.CategoricalCrossentropy(), tf.keras.losses.BinaryCrossentropy()],
                      [self.pred, self.domain], true_pre)]
        return losses

    def build(self):
        self.is_graph_network = True
        self.__init__graph_network(inputs=self.input_layer,
                                   outputs=self.out)