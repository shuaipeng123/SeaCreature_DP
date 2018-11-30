import keras.models
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

batch_size = 128

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

train_gen = train_datagen.flow_from_directory(
    'images/train',
    target_size=(150, 150),
    batch_size=batch_size,
    color_mode='grayscale',
    shuffle=True
)
val_gen = test_datagen.flow_from_directory(
    'images/val',
    target_size=(150, 150),
    batch_size=batch_size,
    color_mode='grayscale',
    shuffle=False
)

model = keras.models.load_model('resnet16_tl.h5')

train_steps = train_gen.samples // batch_size
val_steps = val_gen.samples // batch_size

model.fit_generator(
    train_gen,
    steps_per_epoch=train_steps,
    epochs=10,
    verbose=1,
    validation_data=val_gen,
    validation_steps=val_steps
)
model.save('resnet16_tl.h5')

val_gen = test_datagen.flow_from_directory(
    'images/val',
    target_size=(150, 150),
    batch_size=batch_size,
    shuffle=False
)
predictions = model.predict_generator(val_gen, val_steps)

y_pred = np.argmax(predictions, axis=1)
y_true = val_gen.classes[0:len(y_pred)]

print(precision_recall_fscore_support(y_true, y_pred, pos_label=None, average='macro'))
print(accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))