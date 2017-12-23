import gamedata as gd
import numpy as np
import tensorflow as tf


def train_policy(policy_estimator, game_dataset, num_epochs=1000, batch_size=128, session=None, summary_writer=None, save_path=None):
    # Run supervised training for specified epochs and batch size.
    dataset = gd.GameDataSet(game_dataset)

    # shuffle the data
    shuffle_indices = np.random.permutation(np.arange(len(dataset.y_data)))
    dev_sample_percentage = .1
    dev_sample_index = -1 * int(dev_sample_percentage * float(len(dataset.y_data)))
    x_train, x_val = dataset.x_data[shuffle_indices[:dev_sample_index]], dataset.x_data[shuffle_indices[dev_sample_index:]]
    y_train, y_val = dataset.y_data[shuffle_indices[:dev_sample_index]], dataset.y_data[shuffle_indices[dev_sample_index:]]

    print('pre-training...')

    # create train and test data iterators
    tr = gd.DataIterator(x_train, y_train)

    step, total_accuracy, total_loss = 0, 0, 0
    tr_losses, te_losses = [], []
    current_epoch = 0

    # iterate for 10 epochs
    while current_epoch < num_epochs:

        step += 1

        # get next batch of training data and run a single training iteration
        x_batch, y_batch = tr.next_batch(batch_size)
        loss = policy_estimator.train(x_batch, y_batch)

        if policy_estimator.global_step % 10 == 0:
            accuracy = policy_estimator.training_accuracy(x_batch, y_batch)
            print('iter {} accuracy {} loss {}'.format(policy_estimator.global_step, accuracy, loss))
            if summary_writer is not None:
                policy_estimator.write_summaries(summary_writer)

            # Save the weights
            if save_path is not None:
                if session is None: session = tf.get_default_session()
                saver = tf.train.Saver()
                saver.save(session, save_path)

        # accumulate training accuracy and loss
        total_accuracy += policy_estimator.training_accuracy(x_batch, y_batch)
        total_loss += loss

        # print training progress and run model validation every epoch
        if tr.epochs > current_epoch:

            current_epoch += 1
            tr_losses.append((total_accuracy / step, total_loss / step))
            step, total_accuracy, total_loss = 0, 0, 0

            # eval test set
            te = gd.DataIterator(x_val, y_val, False)
            te_epoch = te.epochs
            while te.epochs == te_epoch:
                step += 1
                x_batch, y_batch = te.next_batch(batch_size)
                total_accuracy += policy_estimator.training_accuracy(x_batch, y_batch)
                total_loss += policy_estimator.training_loss(x_batch, y_batch)

            te_losses.append((total_accuracy / step, total_loss / step))
            step, total_accuracy, total_loss = 0, 0, 0
            print('After epoch {0} (Accuracy, Loss) - tr: ({1:.4f}, {2:.4f}) - te: ({3:.4f}, {4:.4f})'.format(
                    current_epoch, tr_losses[-1][0], tr_losses[-1][1], te_losses[-1][0], te_losses[-1][1]))

