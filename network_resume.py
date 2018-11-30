import numpy as np
import keras.models
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

model = keras.models.load_model('resnet16_tl.h5')

batch_size = 64

train_datagen = ImageDataGenerator(
    rotation_range=360,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.35,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator()

#train_gen = train_datagen.flow_from_directory(
#    'images/train',
#    target_size=(224, 224),
#    batch_size=batch_size,
#    shuffle=True
#)
#val_gen = test_datagen.flow_from_directory(
#    'images/val',
#    target_size=(224, 224),
#    batch_size=batch_size,
#    shuffle=False
#)
test_gen = test_datagen.flow_from_directory(
    'images/test',
    target_size=(150,150),
    batch_size=batch_size,
    color_mode="grayscale",
    shuffle=False
)

#train_steps = train_gen.samples // batch_size
#val_steps = val_gen.samples // batch_size
test_steps = test_gen.samples // batch_size + 1

#model.fit_generator(
#    train_gen,
#    steps_per_epoch=train_steps,
#    epochs=1,
#    verbose=1,
#    validation_data=val_gen,
#    validation_steps=val_steps
#)
#model.save('resnet50.h5')

#val_gen = test_datagen.flow_from_directory(
#    'images/val',
#    target_size=(224, 224),
#    batch_size=batch_size,
#    shuffle=False
#)
predictions = model.predict_generator(test_gen, test_steps)

y_pred = np.argmax(predictions, axis=1)
y_true = test_gen.classes[0:len(y_pred)]

print(precision_recall_fscore_support(y_true, y_pred, pos_label=None, average='macro'))
print(accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))
