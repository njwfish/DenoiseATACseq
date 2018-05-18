from random import randint

import keras
from keras.layers import Conv2D, Dense, Reshape
from net_morphisms import Morphisms

from keras_utils import save_model
from nash.snapshot import SnapshotCallbackBuilder


class NASH(object):
    def __init__(self, steps, children, epochs, X_train, y_train):
        self.steps = steps
        self.children = children
        self.epochs = epochs
        self.X_train = X_train
        self.y_train = y_train
        self.morph = Morphisms()

    def search(self, baseline_net, save_nets=0):
        """
        Greedy hill search for nets based on validation accuracy.
        This is an implementatino of the technique validated here: https://arxiv.org/pdf/1711.04528.pdf
        :param baseline_net: the starting parent network
        :param save_nets: save intermediate nets or not
        :return:
        """
        parent = baseline_net
        parent.compile(loss="mse", optimizer="adam", metrics=["mse","mae"])
        parent, loss = self.train_cosine_annealing(parent, epochs=self.epochs)
        for i in range(self.steps):
            nets = [(loss, parent)]
            modifiable_layers = []
            # Create list of layers that can be modified
            for j in range(len(parent.layers) - 1):  # Cannot modify last layer
                if isinstance(parent.layers[j], (Conv2D, Dense)):
                    # NASH cannot modify 'bridge' layers where model transitions from Dense to Conv or vice versa
                    if j != len(parent.layers) - 1 and isinstance(parent.layers[j + 1], Reshape):
                        continue
                    modifiable_layers.append(j)
            possible_moves = self.gen_possible_moves(modifiable_layers)
            # create and train variations of parent according to self.children, do not exceed possible children (moves)
            for _ in range(min(self.children, len(possible_moves))):
                child, possible_moves = self.gen_child(parent, possible_moves)
                child, loss = self.train_cosine_annealing(child, epochs=self.epochs)
                nets.append((loss, child))
            # select net with lowest validation loss to be new parent
            loss, parent = min(nets, key=lambda x: x[0])
            if save_nets == 1:
                save_model("model_" + str(i) + str(loss), parent, None)
        return parent

    def gen_possible_moves(self, l, widen_range=(3,6), kernel_range=(2,5), skip_types=2):
        """
        Generate all possible
        :param l: the number of layers
        :return: moves: a list of tuples of all possible moves
            moves[0]: move type, deepen, widen, skip
            moves[1]:

        """
        widen = [(0, i, j) for i in l for j in range(widen_range[0], widen_range[1])]
        deepen = [(1, i, j) for i in l for j in range(kernel_range[0], kernel_range[0])]
        # Skip connections not yet implemented
        #skip = [(2, j, i, k) for i in range(1, l - 2) for j in range(i + 2, l) for k in range(skip_types)]
        return widen + deepen #+ skip

    def widened_layer(self, move, parent):
        """
        Create a widened layer, update weight inthat and next layer accordingly
        :param move: he details of the move being made
        :param parent: the parent model
        :return: the modified widened layer and the following layer with updated weights
        """
        move_type, layer, new_width = move
        W1, b1 = parent.layers[layer].get_weights()
        W2, b2 = parent.layers[layer + 1].get_weights()
        if W1.ndim == 2:
            student_w1, student_b1, student_w2 = self.morph.widen(W1, b1, W2, W1.shape[1] + new_width)
            layer = Dense(student_w1.shape[1], weights=[student_w1, student_b1], name="dense_one")
            layer_n = Dense(student_w2.shape[1], weights=[student_w2, b2], name="dense_one")
        else:
            student_w1, student_b1, student_w2 = self.morph.widen(W1, b1, W2, W1.shape[3] + new_width)
            layer = Conv2D(student_w1.shape[3], (student_w1.shape[0], student_w1.shape[1]),
                           padding="same", weights=[student_w1, b1])
            layer_n = Conv2D(student_w2.shape[3], (student_w2.shape[0], student_w2.shape[1]),
                           padding="same", weights=[student_w2, b2], name="conv2d_one")
        # Ensure no name conflicts for keras; in future this will be less bad
        layer.name = 'added' + str(abs(hash(str(layer))))
        layer_n.name = 'added' + str(abs(hash(str(layer_n))))
        return layer, layer_n

    def deepened_layer(self, move, parent):
        """
        Return new layer with weights and bias set to not affect prediction
        :param move: the details of the move being made
        :param parent: the parent model
        :return: the layer to add to the child model, based on the move
        """
        move_type, layer, new_width = move
        W = parent.layers[layer].get_weights()[0]
        # make this take kernel size into account
        W_deeper, b_deeper = self.morph.deepen(W)
        if W_deeper.ndim == 2:
            layer = Dense(W_deeper.shape[1], weights=[W_deeper, b_deeper])
        else:
            layer = Conv2D(W_deeper.shape[3], (W_deeper.shape[0], W_deeper.shape[1]),
                                        weights=[W_deeper, b_deeper], padding="same")
        return layer

    # Currently architecture search does not use skip connections, this will be implemented later
    #def skip_layer(self, move, parent):


    def gen_child(self, parent, possible_moves):
        """
        Create child network using Keras Model API
        :param parent: parent model
        :param possible_moves: list of remaining possible moves
        :return: child model which is a variation of parent according to a selected move, and updated possible moves
        """
        m = randint(0, len(possible_moves) - 1)
        move = possible_moves[m]
        del possible_moves[m]
        input_tensor = parent.layers[0].get_output_at(0)
        Z = input_tensor
        ori_to_new = {str(l.get_output_at(0)): l.get_output_at(0) for l in parent.layers}
        out = []
        inp = []
        out.append(input_tensor)
        inp.append(None)
        # Build lists of original parent inputs and ouputs for each layer,
        for l in parent.layers[1:]:
            # Not set up for layers with multiple inputs
            l_inp = l.get_input_at(0)
            if not isinstance(l_inp, list):
                ori_to_new.update({str(l_inp): out[-1]})
            l_out = l.get_output_at(0)
            ori_to_new.update({str(l_out): l_out})
            out.append(l_out)
            inp.append(l_inp)
        W_next = None

        # Rebuild parent net, layer by layer, substituting in for variations according to move, preserving weights
        for i in range(1, len(parent.layers)):
            l = parent.layers[i]
            ori_in = inp[i]
            ori_out = out[i]
            if isinstance(ori_in, list):
                new_in = [ori_to_new[str(i)] for i in ori_in]
            else:
                # It is unclear why this is necessary, but it is. Removing this try except causes errors. This is odd.
                try:
                    new_in = ori_to_new[str(ori_in)]
                except:
                    new_in = ori_to_new[str(ori_in)]

            if W_next is not None:
                l = W_next
                W_next = None

            if move[1] == i:
                if move[0] == 0:
                    l, W_next = self.widened_layer(move, parent)
                if move[0] == 1:
                    l_added = self.deepened_layer(move, parent)
                    Z = l_added(new_in)
                    new_in = Z
                if move[0] == 2:
                    continue

            # It is unclear why this is necessary, but it is. Removing this try except causes errors. This is odd.
            try:
                Z = l(new_in)
            except:
                Z = l(new_in)

            ori_to_new.update({str(ori_out): Z})
            ori_to_new.update({str(Z): Z})
        # Compile and train child model
        child = keras.models.Model(inputs=input_tensor, outputs=Z)
        child.compile(loss="mse", optimizer="adam", metrics=["mse","mae"])
        return child, possible_moves

    def train_cosine_annealing(self, model, epochs=20, m=5, alpha_zero=0.1):
        """
        Train model with cosine annealing
        :param model: network to train
        :param epochs: number of epochs to train for
        :param m: param for cosine annealing, use default from paper
        :param alpha_zero: param for cosine annealing, use default from paper
        :return: trained model and validation loss
        """
        # Use snapshots to save best models at intermediate milestones
        snapshot = SnapshotCallbackBuilder(epochs, m, alpha_zero)
        h = model.fit(self.X_train, self.y_train, validation_split=0.05,
                      callbacks=snapshot.get_callbacks(model_prefix="tmp"))
        return model, h.history['val_loss'][-1]
