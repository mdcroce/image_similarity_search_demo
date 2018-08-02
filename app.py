import os
import numpy as np
import uuid
import time
import hashlib
import datetime
import logging
import flask
import werkzeug
import argparse
import tornado.wsgi
import tornado.httpserver
import urllib.request
import sklearn.metrics.pairwise
from PIL import Image
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import h5py
import urllib
import exifutil
import sys
sys.path.insert(0, "models/research")
sys.path.insert(0, "models/research/slim")
from feature_extractor import FeatureExtractor

RESULTS = 10

REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../..')
UPLOAD_FOLDER = '/tmp/matheuzin_ceara'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

# Obtain the flask app object
app = flask.Flask(__name__)


def compare_to_base(app, features):
    distances = sklearn.metrics.pairwise.pairwise_distances(app.base, Y=features, metric='cosine', n_jobs=1)
    results = np.argsort(distances, axis=0)[:RESULTS].squeeze()
    scores = [float(score) for score in distances[results]]
    names = [os.path.basename(str(name)[:-4]) for name in app.filenames[results]]
    ids = list(zip(names, scores))
    return ids


def hash_url(url):
    # uuid is used to generate a random number
    salt = uuid.uuid4().hex
    return hashlib.sha256(salt.encode() + url.encode()).hexdigest()

@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)


@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        imagefile = os.path.join(UPLOAD_FOLDER, hash_url(imageurl) + ".jpg")
        img = Image.open(urllib.request.urlopen(imageurl))
        img.save(imagefile)
    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )

    logging.info('Image: %s', imageurl)

    app.clf.enqueue_image_files([imagefile])
    starttime = time.time()
    predicted = app.clf.feed_forward_batch(app.layer_names)
    endtime = time.time()
    timed = '%.3f' % (endtime - starttime)
    features = predicted[app.layer_names[0]].squeeze().reshape(1, -1)
    result = compare_to_base(app, features)
    return flask.render_template(
        'index.html', has_result=True, timed=timed, result=(True, result), imagesrc=imageurl)


@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
        image = exifutil.open_oriented_im(filename)

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    # result = app.clf.classify_image(image)

    app.clf.enqueue_image_files([imagefile])
    starttime = time.time()
    predicted = app.clf.feed_forward_batch(app.layer_names)
    endtime = time.time()
    timed = '%.3f' % (endtime - starttime)
    features = predicted[app.layer_names[0]]
    result = compare_to_base(features)

    return flask.render_template(
        'index.html', has_result=True, result=(True, result), timed=timed,
        imagesrc=embed_image_html(image)
    )


def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )

def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = argparse.ArgumentParser(description="TensorFlow feature extraction")
    parser.add_argument("--port", dest="port", type=int, default=8080, help="Port number")
    args = parser.parse_args()

    feature_extractor = FeatureExtractor(
                                network_name="resnet_v1_101",
                                checkpoint_path="checkpoints/resnet_v1_101.ckpt",
                                batch_size=1,
                                num_classes=1000,
                                preproc_func_name=None,
                                preproc_threads=1)

    # Initialize classifier + warm start by forward for allocation
    app.clf = feature_extractor
    app.layer_names = ["resnet_v1_101/logits"]
    h5 = h5py.File("features_test.h5", "r")
    app.base = np.array(h5['resnet_v1_101']['logits']).squeeze()
    app.filenames = np.array(h5["filenames"])
    app.run(debug=True, host='0.0.0.0', port=args.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)
