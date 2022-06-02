from flask import Flask, current_app
import interv_demo.app

app = interv_demo.app.app
# app.init_db()
# # current_app.init_db()
#app.app.run(debug=True, host="0.0.0.0")

if __name__ == "__main__":
        app.run(debug=True, host="0.0.0.0", port=5001)
