from flask import Flask, request, jsonify
import torch
from train_transformer import TransformerModel
from integrate_llm import llm_enhance

app = Flask(__name__)

MODEL_PATH = 'transformer_model.pth'
INPUT_DIM = 14

if not torch.cuda.is_available():
    print("Warning: Running on CPU — performance may be limited.")

model = TransformerModel(input_dim=INPUT_DIM)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    print(f"Model loaded successfully from {MODEL_PATH}")

except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

@app.route('/')
def home():
    return "Turbofan Predictive Maintenance API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()

    if 'seq' not in data or 'engine_id' not in data:
        return jsonify({"error": "Missing required fields: 'seq' and 'engine_id'"}), 400
    
    try:
        seq_list = data['seq']
        if not isinstance(seq_list, list) or len(seq_list) != 50:
            return jsonify({"error": "Sequence must be a list of 50 timesteps"}), 400
        
        seq = torch.tensor(seq_list, dtype=torch.float32)

        if seq.shape[1] != INPUT_DIM:
            return jsonify({"error": f"Each timestep must have {INPUT_DIM} features"}), 400
        
        # Add batch dimension
        seq = seq.unsqueeze(0)

        engine_id = str(data['engine_id'])

        # Forward pass
        with torch.no_grad():
            anom_out, rul_out = model(seq)
            anom_prob = torch.softmax(anom_out, dim=1)[0][1].item()
            rul_est = rul_out.item()

            # LLM Refinement
            enhanced = llm_enhance(seq.squeeze(0), engine_id, anom_prob, rul_est)
            return jsonify({
                "anomaly_probability": round(anom_prob, 4),
                "estimated_rul": round(rul_est, 1),
                "enhanced_prediction": enhanced
            })

    except ValueError as ve:
        return jsonify({"error": f"Invalid input format: {str(ve)}"}), 400
    
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == '__main__':
    print("Starting Predictive Maintenance API...")
    app.run(debug=True, host='0.0.0.0', port=5001)