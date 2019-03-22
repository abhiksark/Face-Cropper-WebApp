from flask import Flask, redirect, render_template, request, session, url_for
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES
from flask_uploads import patch_request_class

import os
import facenet
import tensorflow as tf
import detect_face
import numpy as np
from scipy import misc

app = Flask(__name__ , static_url_path='/faces/')
dropzone = Dropzone(app)


app.config['SECRET_KEY'] = 'secretkey'

# Dropzone settings
app.config['DROPZONE_UPLOAD_MULTIPLE'] = False
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'results'

# Uploads settings
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/uploads/photos'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB


@app.route('/', methods=['GET', 'POST'])
def index():
    file_urls = os.listdir("./uploads/photos/")
    for image in file_urls:
        os.remove("./uploads/photos/" + image)
    # set session for image results
    if "file_urls" not in session:
        session['file_urls'] = []
    # list to hold our uploaded image urls
    file_urls = session['file_urls']

    # handle image upload from Dropszone
    if request.method == 'POST':
        file_obj = request.files
        for f in file_obj:
            file = request.files.get(f)            
            # save the file with to our photos folder
            filename = photos.save(
                file,
                name=file.filename
            )

            # append image urls
            file_urls.append(photos.url(filename))  
        print(file_urls)          
        session['file_urls'] = file_urls
        return "uploading..."
    # return dropzone template on GET request
    return render_template('index.html')


@app.route('/results')
def results():
    datadir = './uploads/'
    output_dir_path = './faces/'
    output_dir = os.path.expanduser(output_dir_path)
    dataset = facenet.get_dataset(datadir)

    with tf.Graph().as_default():
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        sess = tf.Session()
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, './data')

    minsize = 40  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor 0.709
    margin = 44
    image_size = 182

    # Add a random key to the filename to allow 
    # alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)

    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        for cls in dataset:
            output_class_dir = os.path.join(output_dir, cls.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
            for image_path in cls.image_paths:
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir, filename + '.jpg')
                print(image_path)
                if not os.path.exists(output_filename):
                    try:
                        img = misc.imread(image_path)
                        print('read data dimension: ', img.ndim)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim < 2:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                            print('to_rgb data dimension: ', img.ndim)
                        img = img[:, :, 0:3]
                        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                        nrof_faces = bounding_boxes.shape[0]
                        counter = nrof_faces
                        det = bounding_boxes[:, 0:4]
                        img_size = np.asarray(img.shape)[0:2]
                        
                        if nrof_faces == 1:
                            det = np.squeeze(det)
                            bb_temp = np.zeros(4, dtype=np.int32)
                            bb_temp[0] = det[0]
                            bb_temp[1] = det[1]
                            bb_temp[2] = det[2]
                            bb_temp[3] = det[3]
                            try: 
                                cropped_temp = img[bb_temp[1]:bb_temp[3], bb_temp[0]:bb_temp[2], :]
                                scaled_temp = misc.imresize(cropped_temp, (image_size, image_size), interp='bilinear')
                            except (ValueError) as e:
                                print("No Print")
                                continue
                            nrof_successfully_aligned += 1
                            misc.imsave(output_filename, scaled_temp)
                            text_file.write('%s %d %d %d %d\n' % (output_filename, bb_temp[0], bb_temp[1], bb_temp[2], bb_temp[3]))

                        if nrof_faces > 1:
                                bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                                img_center = img_size / 2
                                offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                    (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                                index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                                # det = det[index, :]
                                det_morethanone = det
                                for i in range(nrof_faces):
                                    det = det_morethanone[int(i),:]
                                    det = np.squeeze(det)
                                    bb_temp = np.zeros(4, dtype=np.int32)
                                    bb_temp[0] = det[0]
                                    bb_temp[1] = det[1]
                                    bb_temp[2] = det[2]
                                    bb_temp[3] = det[3]
                                    try:
                                        cropped_temp = img[bb_temp[1]:bb_temp[3], bb_temp[0]:bb_temp[2], :]
                                        scaled_temp = misc.imresize(cropped_temp, (image_size, image_size), interp='bilinear')
                                    except (ValueError) as e:
                                        print("No Print")
                                        continue
                                    nrof_successfully_aligned += 1
                                    misc.imsave(output_filename[:-4]+str(i)+"1.jpg", scaled_temp)
                                    text_file.write('%s %d %d %d %d\n' % (output_filename, bb_temp[0], bb_temp[1], bb_temp[2], bb_temp[3]))
                        else:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))

    # redirect to home if no images to display
    if "file_urls" not in session or session['file_urls'] == []:
        return redirect(url_for('index'))
        
    # set the file_urls and remove the session variable
    file_urls = os.listdir("./faces/photos/")
    
    for image in file_urls:
        print("/faces/photos/"+image)
        print("./uploads/photos/"+image)
        os.rename("./faces/photos/"+image ,"./uploads/photos/"+image)
    dir = "_uploads/photos/"
    file_urls = [dir + x for x in file_urls]
    
    
    # file_urls = session['file_urls']
    print(file_urls)
    session.pop('file_urls', None)
    
    return render_template('results.html', file_urls=file_urls)
