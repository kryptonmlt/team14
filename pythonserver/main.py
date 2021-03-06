from flask import Flask
from flask import request

import os
from flask import Flask, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename

import pictureprocess

UPLOAD_FOLDER = './'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.secret_key = 'super secret key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

from datetime import timedelta
from flask import make_response, request, current_app
from functools import update_wrapper

## make the reply cross domain
def crossdomain(origin=None, methods=None, headers=None,
                max_age=21600, attach_to_all=True,
                automatic_options=True):
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, basestring):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, basestring):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers

            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)

    return decorator


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

## POST the game image and return the players and walls locations
@app.route('/upload', methods=['GET', 'POST'])
@crossdomain(origin='*')
def upload_file():
    if request.method == 'POST':

        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            print('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # compute the stuff
            p1, p2, wall_tileids, pxlimg, orgimg1, background_tile = pictureprocess.create_world(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            print background_tile

            # picture_url = url_for('uploaded_file', filename=filename)
            picture_url = filename

            import json
            result = json.dumps({'p1':p1, 'p2':p2,
                                # 'objs':','.join([str(x) for x in list(wall_tileids)]),
                                'background_tile':str(background_tile),
                                'objs':'wall_ids.txt',
                                'pictureUrl':picture_url}, separators=(',', ':'))

            ## save the pixelated image
            import scipy
            from scipy import misc
            misc.imsave(os.path.join(app.config['UPLOAD_FOLDER'], filename), orgimg1)

            ## save the walls position
            with open('./wall_ids.txt', 'w') as f:
                d_ = ','.join([str(i) for i in wall_tileids.reshape(-1)])
                f.write(d_)

            return result;

    return 'ok'


@app.route('/uploads/<filename>')
@crossdomain(origin='*')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == "__main__":
    app.run()
