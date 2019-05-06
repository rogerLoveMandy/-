
from keras.models import   load_model

from load_face_dataset import load_dataset, resize_image, IMAGE_SIZE

model = load_model("model_weight.h5")
print(model)

images,_= load_dataset('D:/tmp/output/')

image = images.reshape(images.shape[0], 64, 64, 3)
result = model.predict_classes(image)

print ('Predicted:', result)



