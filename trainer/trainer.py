from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

import tensorflow as tf




# img_width, img_height = 150, 150


train_data_dir = 'gs://bucket_groep1/train/'
test_data_dir = 'gs://bucket_groep1/test/'

#3725
#928
nb_train_samples = 3725
nb_validation_samples = 928
nb_epoch = 20
nb_batch_size = 16
img_width=299
img_height=299


# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(9, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


#prepare data augmentation configuration

train_datagen = ImageDataGenerator(
        rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=nb_batch_size,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=nb_batch_size,
    class_mode='categorical'
)

print("start history model")
history = model.fit_generator(
    train_generator,
    nb_epoch=nb_epoch,
    samples_per_epoch=nb_train_samples,
    validation_data=validation_generator,
    nb_val_samples=58, workers=1)



#evaluatie van het model
score = model.evaluate_generator(validation_generator, nb_validation_samples, workers=1)
print('accuracy: {0}'.format(score))


K.set_learning_phase(0)  # test
sess = K.get_session()

from tensorflow.python.framework import graph_util

# Make GraphDef of Transfer Model
g_trans = sess.graph
g_trans_def = graph_util.convert_variables_to_constants(sess,
                                                        g_trans.as_graph_def(),
                                                        [model.output.name.replace(':0','')])

# Image Converter Model
with tf.Graph().as_default() as g_input:
    input_b64 = tf.placeholder(shape=(1,), dtype=tf.string, name='input')
    input_bytes = tf.decode_base64(input_b64[0])
    image = tf.image.decode_image(input_bytes)
    image_f = tf.image.convert_image_dtype(image, dtype=tf.float32)
    input_image = tf.expand_dims(image_f, 0)
    output = tf.identity(input_image, name='input_image')

g_input_def = g_input.as_graph_def()



with tf.Graph().as_default() as g_combined:
    x = tf.placeholder(tf.string, name="input_b64")

    im, = tf.import_graph_def(g_input_def,
                              input_map={'input:0': x},
                              return_elements=["input_image:0"])

    pred, = tf.import_graph_def(g_trans_def,
                                input_map={model.input.name: im,
                                          'batch_normalization_1/keras_learning_phase:0': False},
                                return_elements=[model.output.name])

    with tf.Session() as sess2:
        inputs = {"inputs": tf.saved_model.utils.build_tensor_info(x)}
        outputs = {"outputs": tf.saved_model.utils.build_tensor_info(pred)}
        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )

        # save as SavedModel
        b = tf.saved_model.builder.SavedModelBuilder('gs://bucket_groep1/model3')
        b.add_meta_graph_and_variables(sess2,
                                       [tf.saved_model.tag_constants.SERVING],
                                       signature_def_map={'serving_default': signature})
        b.save()

