import warnings

with warnings.catch_warnings():
    import os, sys
    import cv2
    import tensorflow as tf
    import keras
    from utils import *
    import numpy as np
    from models import config
    from models import model as M
    from PIL import Image, ImageDraw
    from keras.models import Model
    from keras.preprocessing.image import array_to_img

C = config.Config()

def get_shape(img, w, h, scale, ratio):

    size = (w, h)
    if ratio:
        if scale is not None:
            size = (scale, scale)
    else:
        if scale is not None:
            if w <= h:
                size = (scale, scale * h // w)
            else:
                size = (scale * w // h, scale)
    return img.resize(size, Image.ANTIALIAS)


def generate_saliency_map(model, images):
    if C.log:
        result = []
    if os.path.isdir(images):
        test_db = os.listdir(images)
        test_db = [os.path.join(images, i) for i in test_db]
    elif os.path.isfile(images) and images.endswith(C.pic_extend):
        test_db = list(images)
    else:
        raise Exception('Image file or directory not exist.')
    for image_name in test_db:

        if not image_name.endswith(C.pic_extend):
            continue
        img_object = Image.open(image_name)
        w3, h3 = img_object.size
        img_object = img_object.convert('RGB')
        #imgs = img.copy()
        img_reshape = get_shape(img_object, w3, h3, C.scale, C.ratio)
        image = np.asarray(img_reshape)

        h1, w1 = image.shape[0], image.shape[1]
        

        h2, w2 = (image.shape[0] // 16 + 1) * 16, (image.shape[1] // 16 + 1) * 16
        image = cv2.copyMakeBorder(image, top=0, bottom=h2 - h1, left=0, right=w2 - w1, borderType=cv2.BORDER_CONSTANT,
                                   value=0)
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)

        model = Model(inputs=model.input, outputs=model.get_layer('segmentation').output)
        saliency_out = model.predict(image, batch_size=1, verbose=0)

        # get saliency map here
        saliency_map = np.array(saliency_out)[0].squeeze(-1)
        img_name = image_name.split('/')[-1]

        if not os.path.isdir(C.saliency_out_path):
            os.makedirs(C.saliency_out_path)
        saliency_map = (((saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())) * 255.9).astype(np.uint8)
        saliency_im = Image.fromarray(saliency_map)
        saliency_im.save(os.path.join(C.saliency_out_path, img_name))

def main(argv=None):

    if len(sys.argv)<=1:
        images = C.image_path
    else:
        images = sys.argv[1]
    model = M.EndToEndModel(gamma=C.gamma, theta=C.theta, stage='test').BuildModel()
    model.load_weights(C.model)
    print('weights loaded successfully!')
    generate_saliency_map(model, images)

if __name__ == "__main__":
    sys.exit(main())
