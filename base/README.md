A typical usage of models is as following:
########################### ResNet ################################
    from ResNet import ResNet_forward
    inPuts = ...(a tf.placeholder)
    outPuts = ResNet_forward(inPuts=inPuts,
                             resnet_size=50,
                             is_classification=False,
                             num_classes=None,
                             is_train=False,
                             data_format='channels_last')
    ## model variable scope: 'resnet_model'
    ## The way to restore weights in this model is:
        # saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_model'))
        # saver.restore(sess, weight_path)


########################### Xception ################################
    from Xception import Xception_forward
    inPuts = ...(a tf.placeholder)
    endPoints = Xception_forward(inPuts=inPuts,
                             xception_size=41,
                             is_classification=False,
                             num_classes=None,
                             is_train=False,
                             data_format='channels_last')
    ### model variable scope: 'xception_41'
    ### endPoints is a dict, ['tensor name': tensor_value]
    ## The way to restore weights in this model is:
        # saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='xception_41'))
        # saver.restore(sess, weight_path)