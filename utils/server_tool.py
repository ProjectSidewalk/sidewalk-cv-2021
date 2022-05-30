import argparse
import http.server
import io
import os
import pandas as pd
import posixpath
import urllib
from ast import literal_eval
from http import HTTPStatus

# Instructions for use:
# 1. If ssh-ing, use "ssh -L local_port:remote_ip:remote_port user@hostname.com".
#    For example, "gcloud compute ssh cv-main --project=computer-vision-344418 --zone=us-central1-b -- -NL 31415:localhost:31415".
#    Make sure the port address matches the port number specified in the command line.
# 2. Run server_tool.py.
# 3. Navigate to "http://localhost:31415/crop_viewer" in your browser and begin editing!
#    Click through crops with "Next" and "Prev". 
#    Clicking "Save" saves and goes to the next crop.
# 4. Use Ctrl+C to exit.

parser = argparse.ArgumentParser()
parser.add_argument('csv_path', type=str, help="csv_path - path to the dataset csv")
parser.add_argument('crops_dir', type=str, help="crops_dir - path to directory containing crop jpgs")
parser.add_argument('port', type=int, help="port - port from which to serve the tool from")
parser.add_argument('crop_size', type=int, help="crop_size - diameter of bounding box to apply to crops")
args = parser.parse_args()

def generate_label_ids_to_csv_indices_map(csv_df):
  filenames = csv_df['image_name']
  label_ids_to_csv_indices = dict()
  for index, filename in filenames.items():
    # different filename formats that are accepted by search tool
    acceptable_filenames = [filename]
    # without extension
    acceptable_filenames.append(filename[:filename.index(".jpg")]) 
    # without _0 size indicator at end or extension
    if "_0" in filename:
      acceptable_filenames.append(filename[:filename.index("_0")])
    if "/" in filename:
      # without city prefix or extension
      acceptable_filenames.append(filename[filename.index("/") + 1:filename.index(".jpg")])
      # without city prefix or _0 size indicator at end or extension
      acceptable_filenames.append(filename[filename.index("/") + 1:filename.index("_0") if "_0" in filename else len(filename)])

    for name in acceptable_filenames:
      label_ids_to_csv_indices[name] = index

  return label_ids_to_csv_indices

def save_to_file():
  csv_df.to_csv(args.csv_path, index=False)

class MyHTTPHandler(http.server.SimpleHTTPRequestHandler):

  def __init__(self, *args):
    http.server.SimpleHTTPRequestHandler.__init__(self, *args)

  def translate_path(self, path):
    path = path.split("?",1)[0]
    path = path.split("#",1)[0]
    trailing_slash = path.rstrip().endswith("/")
    try:
        path = urllib.parse.unquote(path, errors="surrogatepass")
    except UnicodeDecodeError:
        path = urllib.parse.unquote(path)
    path = posixpath.normpath(path)
    words = path.split("/")
    words = filter(None, words)
    path = args.crops_dir
    for word in words:
        if os.path.dirname(word) or word in (os.curdir, os.pardir):
            continue
        path = os.path.join(path, word)
    if trailing_slash:
        path += "/"
    return path

  def send_head(self):
    global csv_df
    global label_ids_to_csv_indices
    if "/save/" in self.path:
      index = int(self.path.replace("/save/", "").partition("?")[0])
      form_data = self.path.partition("?")[2]
      if form_data != "":
        # parse selected label ids into list
        label_ids = list(map(int, form_data.replace("=on","").split("&")))
      else:
        label_ids = []
      label_ids += filter(lambda label_id: label_id >= 5, csv_df.at[index, 'label_set'])
      csv_df.at[index, 'label_set'] = label_ids
      save_to_file()
      self.send_response(HTTPStatus.MOVED_PERMANENTLY)
      self.send_header("Location", "/crop_viewer/" + str(index + 1))
      self.send_header("Cache-Control", "no-store")
      self.send_header("Content-Length", "0")
      self.end_headers()
      return None
    if "/delete/" in self.path:
      index = int(self.path.replace("/delete/", "").partition("?")[0])
      csv_df = csv_df.drop(index).reset_index(drop=True)
      label_ids_to_csv_indices = generate_label_ids_to_csv_indices_map(csv_df)
      save_to_file()
      self.send_response(HTTPStatus.MOVED_PERMANENTLY)
      self.send_header("Location", "/crop_viewer/" + str(index if index < len(csv_df) else index - 1))
      self.send_header("Cache-Control", "no-store")
      self.send_header("Content-Length", "0")
      self.end_headers()
      return None
    elif self.path == "/crop_viewer" or self.path == "/crop_viewer/":
      self.send_response(HTTPStatus.MOVED_PERMANENTLY)
      self.send_header("Location", "/crop_viewer/0")
      self.send_header("Content-Length", "0")
      self.end_headers()
      return None
    elif "/crop_viewer" in self.path:
      # one-indexed
      if "search=" in self.path:
        searched_label_id = urllib.parse.unquote(self.path[self.path.index("=") + 1:])
        if searched_label_id in label_ids_to_csv_indices.keys():
          index = label_ids_to_csv_indices[searched_label_id]
        else:
          index = 0
      else:
        index = int(self.path.replace("/crop_viewer/", "").partition("?")[0])
      img_id = csv_df.at[index, 'image_name'].replace(".jpg", "")

      image_diameter = 500
      bounding_box_diameter = args.crop_size / 1500 * image_diameter

      text = f"""
      <h2 style="display: flex; justify-content: center; margin-top: 25px;">Crop #{index + 1}/{len(csv_df)}</h2>
      <h3 style="display: flex; justify-content: center;">Image Name: {img_id}</h3>
      <div style="display: flex; justify-content: center;">
        <img src="/{img_id}.jpg" width="{image_diameter}" height="{image_diameter}"></img>
        <div style="
            opacity=0;
            position: absolute;
            margin-top: {(image_diameter - bounding_box_diameter) / 2}px;
            width: {bounding_box_diameter}px;
            height: {bounding_box_diameter}px;
            outline: {(image_diameter - bounding_box_diameter) / 2 + 1}px solid black;
          "></div>
      </div>
      <div style="display: flex; justify-content: center; margin-top: 15px;">
        <a href="/crop_viewer/{index - 1 if index > 0 else index}" style="margin-right: 125px;">Prev</a>
        <form method="get" action="/save/{index}">
          <div>
            <input type="checkbox" id="1" name="1" {"checked" if 1 in csv_df.at[index, 'label_set'] else ""}>
            <label for="1"> Curb Ramp</label><br>
          </div>
          <div>
            <input type="checkbox" id="2" name="2" {"checked" if 2 in csv_df.at[index, 'label_set'] else ""}>
            <label for="2"> Missing Curb Ramp</label><br>
          </div>
          <div>
            <input type="checkbox" id="3" name="3" {"checked" if 3 in csv_df.at[index, 'label_set'] else ""}>
            <label for="3"> Obstacle</label>
          </div>
          <div>
            <input type="checkbox" id="4" name="4" {"checked" if 4 in csv_df.at[index, 'label_set'] else ""}>
            <label for="4"> Surface Problem</label>
          </div>
          <div style="margin-top: 5px;">
            <input type="submit" value="Save">
          </div>
        </form>
        <a href="/crop_viewer/{index + 1 if index < len(csv_df) - 1 else index}" style="margin-left: 125px;">Next</a>
      </div>
      <div style="display: flex; justify-content: center; margin-top: 15px;">
        <a href="/delete/{index}" style="margin-left: 430px;">Delete</a>
      </div>
      <div style="display: flex; justify-content: center; margin-top: 15px;">
        <form action="/crop_viewer">
            Search by image name:
            <input type="text" name="search" id="search">
            <button type="submit" id="go">Go</button>
        </form>
      </div>
      """
      encoded = text.encode("utf-8")
      f = io.BytesIO()
      f.write(encoded)
      f.seek(0)
      self.send_response(HTTPStatus.OK)
      self.send_header("Content-type", "text/html; charset=utf-8")
      self.send_header("Cache-Control", "no-store")
      self.send_header("Content-Length", str(len(encoded)))
      self.end_headers()
      return f
    else:
      return http.server.SimpleHTTPRequestHandler.send_head(self)

csv_df = pd.read_csv(args.csv_path, converters={'label_set': literal_eval})
label_ids_to_csv_indices = generate_label_ids_to_csv_indices_map(csv_df)

Handler = MyHTTPHandler

with http.server.HTTPServer(("", args.port), Handler) as httpd:
  print("Serving at port", args.port)
  httpd.serve_forever()
