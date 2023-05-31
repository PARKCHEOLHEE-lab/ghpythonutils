from flask import Flask, jsonify
from utils import *


app = Flask(__name__)


@app.route("/p2m/<path:image_path_param>")
def p2m(image_path_param):
    """convert plan to model"""
    print(image_path_param)
    
    return jsonify(
        {
            "path": image_path_param
        }
    )



if __name__ == "__main__":  
   app.run("0.0.0.0",port=5000,debug=True)