import keras

class PrintOutputValAccOnly(keras.callbacks.Callback):
    def __init__(self):
        self.best_acc = 0

    def on_epoch_begin(self, epoch, logs={}):
        print("epoch:{}".format(epoch))

    def on_epoch_end(self, batch, logs={}):
        self.best_acc = max(self.best_acc, logs["val_output_noised_acc"])
        print("val_output_acc:{val_output_acc}, val_output_noised_acc:{val_output_noised_acc}, best_acc={best_acc}".format(best_acc=self.best_acc, **logs))


class PrintEvaluate(keras.callbacks.Callback):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train


    def on_epoch_end(self, batch, logs={}):
        evaluate_result = self.model.evaluate(
                self.X_train, 
                self.y_train
                )
        print evaluate_result
        print(
            "train_output_acc:{output_acc}, train_output_noised_acc:{output_noised_acc}".format(
                **dict(zip(self.model.metrics_names, evaluate_result))
            )
        )
        print("val_output_acc:{val_output_acc}, val_output_noised_acc:{val_output_noised_acc}".format(**logs))

