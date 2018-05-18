import numpy as np


class Morphisms(object):

    def _conv_deeper(self, W):
        assert W.shape[0] % 2 == 1 and W.shape[1] % 2 == 1, 'Kernel size should be odd'
        W_deeper = np.zeros((W.shape[0], W.shape[1], W.shape[3], W.shape[3]))
        center_h = (W.shape[0] - 1) // 2
        center_w = (W.shape[1] - 1) // 2
        for i in range(W.shape[3]):
            W_deeper[center_h, center_w, i, i] = 1
        b_deeper = np.zeros(W.shape[3])
        return W_deeper, b_deeper


    def _fc_deeper(self, W):
        W_deeper = np.eye(W.shape[1])
        b_deeper = np.zeros(W.shape[1])
        return W_deeper, b_deeper

    def deepen(self, W):
        """
        Add layer
        :param W: W matrix of layer preceeding new layer
            'conv': W.ndim = 4 (kH, kW, InChannel, OutChannel)
            'fc':   W.ndim = 2 (In, Out)
        :return: Weight matrix of added layer
        """

        assert W.ndim == 4 or W.ndim == 2, 'Check W.ndim'
        d = {2: self._fc_deeper,  4: self._conv_deeper}
        W_deeper, b_deeper = d[W.ndim](W)
        return W_deeper, b_deeper


    def normalize(self, W):
        """
        Add a batch norm layer after
        :param W: W: W matrix of layer preceeding new layer
            'conv': W.ndim = 4 (kH, kW, InChannel, OutChannel)
            'fc':   W.ndim = 2 (In, Out)
        :return:
        """
        assert W.ndim == 4 or W.ndim == 2, 'Check W.ndim'
        d = {2: self._fc_deeper, 4: self._conv_deeper}
        W_deeper, b_deeper = d[W.ndim](W)
        return W_deeper, b_deeper


    def _wider_conv(self, teacher_w1, teacher_b1, teacher_w2, new_width):
        rand = np.random.randint(teacher_w1.shape[3], size=(new_width - teacher_w1.shape[3]))
        replication_factor = np.bincount(rand)
        student_w1 = teacher_w1.copy()
        student_w2 = teacher_w2.copy()
        student_b1 = teacher_b1.copy()
        # target layer update (i)
        for i in range(len(rand)):
            teacher_index = rand[i]
            new_W = teacher_w1[:, :, :, teacher_index]
            new_W = new_W[:, :, :, np.newaxis]
            student_w1 = np.concatenate((student_w1, new_W), axis=3)
            student_b1 = np.append(student_b1, teacher_b1[teacher_index])
        # next layer update (i+1)
        for i in range(len(rand)):
            teacher_index = rand[i]
            factor = replication_factor[teacher_index] + 1
            assert factor > 1, 'Error in Net2Wider'
            new_W = teacher_w2[:, :, teacher_index, :] * (1. / factor)
            new_W_re = new_W[:, :, np.newaxis, :]
            student_w2 = np.concatenate((student_w2, new_W_re), axis=2)
            student_w2[:, :, teacher_index, :] = new_W
        return student_w1, student_b1, student_w2

    def _wider_fc(self, teacher_w1, teacher_b1, teacher_w2, new_width):
        rand = np.random.randint(teacher_w1.shape[1], size=(new_width - teacher_w1.shape[1]))
        replication_factor = np.bincount(rand)
        student_w1 = teacher_w1.copy()
        student_w2 = teacher_w2.copy()
        student_b1 = teacher_b1.copy()
        # target layer update (i)
        for i in range(len(rand)):
            teacher_index = rand[i]
            new_W = teacher_w1[:, teacher_index]
            new_W = new_W[:, np.newaxis]
            student_w1 = np.concatenate((student_w1, new_W), axis=1)
            student_b1 = np.append(student_b1, teacher_b1[teacher_index])
        # next layer update (i+1)
        for i in range(len(rand)):
            teacher_index = rand[i]
            factor = replication_factor[teacher_index] + 1
            assert factor > 1, 'Error in Net2Wider'
            new_W = teacher_w2[teacher_index, :] * (1. / factor)
            new_W = new_W[np.newaxis, :]
            student_w2 = np.concatenate((student_w2, new_W), axis=0)
            student_w2[teacher_index, :] = new_W
        return student_w1, student_b1, student_w2

    def widen(self, W1, b1, W2, new_width):
        """
        
        """
        # Check dimensions
        assert b1.squeeze().ndim == 1, 'Check bias.ndim'
        assert W1.ndim == 4 or W1.ndim == 2, 'Check W1.ndim'
        assert W2.ndim == 4 or W2.ndim == 2, 'Check W2.ndim'
        b1 = b1.squeeze()
        if W1.ndim == 2:
            assert W1.shape[1] == W2.shape[0], 'Check shape of W'
            assert W1.shape[1] == len(b1), 'Check shape of bias'
            assert W1.shape[1] < new_width, 'new_width should be larger than old width'
            return self._wider_fc(W1, b1, W2, new_width)
        else:
            assert W1.shape[3] == W2.shape[2], 'Check shape of W'
            assert W1.shape[3] == len(b1), 'Check shape of bias'
            assert W1.shape[3] < new_width, 'new_width should be larger than old width'
            return self._wider_conv(W1, b1, W2, new_width)


    def _conv_skip(self, W1, W2):
        W_skip = np.zeros((W2.shape[0], W2.shape[1], W2.shape[2], W2.shape[3] + W1.shape[3]))
        W_skip[:, :, :, :W2.shape[3]] = W2
        return W_skip


    def _fc_skip(self, W1, W2):
        W_skip = np.zeros((W2.shape[0], W2.shape[1] + W1.shape[1]))
        W_skip[:, :W2.shape[1]] = W2
        return W_skip

    def skip(self, W1, W2):
        """

        :param W1: W matrix of layer to skip from
            'conv': W.ndim = 4 (kH, kW, InChannel, OutChannel)
            'fc':   W.ndim = 2 (In, Out)
        :param W2: W matrix of layer to skip to
            'conv': W.ndim = 4 (kH, kW, InChannel, OutChannel)
            'fc':   W.ndim = 2 (In, Out)
        :return: new weight matrix for channel skipping to
        """
        assert W1.ndim == 4, 'Check W1.ndim'
        assert W2.ndim == 4, 'Check W2.ndim'
        W_skip = np.zeros((W2.shape[0], W2.shape[1], W2.shape[2], W2.shape[3] + W1.shape[3]))
        W_skip[:, :, :, :W2.shape[3]] = W2
        return W_skip

