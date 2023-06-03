from flask import Flask, jsonify
from p2m.p2m import P2M


app = Flask(__name__)


@app.route("/p2m/<path:image_path_param>")
def p2m(image_path_param):
    """convert plan to model"""
    
    return jsonify({"wall_coordinates": P2M(image_path_param).wall_coordinates})



if __name__ == "__main__":  
   app.run("0.0.0.0",port=5000,debug=True)