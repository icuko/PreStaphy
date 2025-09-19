from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import os
import importlib
import numpy as np
import pandas as pd
from flask_cors import CORS
from rdkit import Chem  # 确保安装rdkit: pip install rdkit-pypi

# 初始化Flask应用，指定静态文件和模板文件目录
app = Flask(
    __name__,
    static_folder=os.path.dirname(os.path.abspath(__file__)),
    template_folder=os.path.dirname(os.path.abspath(__file__))
)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # 允许跨域请求

# 模型配置
MODEL_CONFIGS = {
    "Model 1": {
        "model_file": "models/model1.joblib",
        "feature_module": "features.model1_feature",
        "default_threshold": 0.94
    },
    "Model 2": {
        "model_file": "models/model2.joblib",
        "feature_module": "features.model2_feature",
        "default_threshold": 0.93
    },
    "Model 3": {
        "model_file": "models/model3.joblib",
        "feature_module": "features.model3_feature",
        "default_threshold": 0.95
    },
    "Model 4": {
        "model_file": "models/model4.joblib",
        "feature_module": "features.model4_feature",
        "default_threshold": 0.94
    },
    "Model 5": {
        "model_file": "models/model5.joblib",
        "feature_module": "features.model5_feature",
        "default_threshold": 0.95
    }
}

models = {}  # 存储加载的模型

# 根路由 - 返回前端页面
@app.route('/')
def serve_index():
    return render_template('index.html')

@app.route('/prediction_worker.js')
def serve_worker():
    # 返回prediction_worker.js文件
    return send_from_directory(
        os.path.dirname(os.path.abspath(__file__)),  # 文件所在目录
        'prediction_worker.js',                     # 文件名
        mimetype='application/javascript'           # 正确的MIME类型
    )
    
# 模型列表接口
@app.route('/api/models')
def get_models():
    """返回可用模型列表及默认阈值"""
    try:
        # 提取模型名称列表
        model_names = list(MODEL_CONFIGS.keys())
        
        # 提取每个模型的默认阈值
        default_thresholds = {
            model: config['default_threshold'] 
            for model, config in MODEL_CONFIGS.items()
        }
        
        return jsonify({
            "models": model_names,
            "default_thresholds": default_thresholds,
            "available": True
        })
    except Exception as e:
        return jsonify({
            "models": [],
            "default_thresholds": {},
            "available": False,
            "error": str(e)
        }), 500

# 特征对齐函数
def align_features_static(current_features, current_names, feature_names_train):
    """将提取的特征与训练时的特征列表对齐"""
    df_current = pd.DataFrame([current_features], columns=current_names)
    for col in feature_names_train:
        if col not in df_current.columns:
            df_current[col] = 0.0
    return df_current[feature_names_train].values[0]

# 验证SMILES有效性
def validate_smiles(smiles):
    """检查SMILES字符串是否有效"""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

# 加载所有模型
def load_all_models():
    """加载配置中的所有模型和对应的特征提取器"""
    for model_name, config in MODEL_CONFIGS.items():
        try:
            # 获取模型文件绝对路径
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, config["model_file"])
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            # 加载模型和训练特征名
            loaded_data = joblib.load(model_path)
            if not isinstance(loaded_data, tuple) or len(loaded_data) != 2:
                raise ValueError(f"模型文件格式错误，需包含模型和特征名元组")
            model, feature_names_train = loaded_data
            
            # 验证模型是否支持概率预测
            if not hasattr(model, 'predict_proba'):
                raise AttributeError(f"模型不支持predict_proba方法")
            
            # 加载特征提取模块
            feature_module = importlib.import_module(config["feature_module"])
            feature_extractor = getattr(feature_module, 'get_all_features', None)
            if not feature_extractor or not callable(feature_extractor):
                raise AttributeError(f"特征模块中未找到可调用的get_all_features函数")
            
            # 存储模型信息
            models[model_name] = {
                "model": model,
                "feature_names": feature_names_train,
                "feature_extractor": feature_extractor,
                "default_threshold": config["default_threshold"]
            }
            print(f"模型 {model_name} 加载成功")
            
        except Exception as e:
            print(f"模型 {model_name} 加载失败: {str(e)}")

# 单模型预测接口
@app.route('/api/predict', methods=['POST'])
def predict():
    """接收SMILES列表并返回预测结果"""
    data = request.json
    
    # 验证输入数据
    if not data:
        return jsonify({"error": "请求数据为空"}), 400
    
    model_name = data.get('model')
    threshold = data.get('threshold')
    smiles_list = data.get('smiles_list', [])
    
    # 验证模型选择
    if not model_name or model_name not in models:
        return jsonify({"error": "无效的模型名称"}), 400
    
    # 使用默认阈值如果未提供
    model_data = models[model_name]
    if threshold is None:
        threshold = model_data["default_threshold"]
    else:
        try:
            threshold = float(threshold)
            if not (0 <= threshold <= 1):
                return jsonify({"error": "阈值必须在0-1之间"}), 400
        except ValueError:
            return jsonify({"error": "阈值必须是数字"}), 400
    
    # 验证SMILES列表
    if not isinstance(smiles_list, list) or len(smiles_list) == 0:
        return jsonify({"error": "请提供有效的SMILES列表"}), 400
    
    # 处理每个SMILES
    results = []
    model = model_data["model"]
    feature_names_train = model_data["feature_names"]
    feature_extractor = model_data["feature_extractor"]
    
    for smi in smiles_list:
        result = {"smiles": smi}
        try:
            # 验证SMILES
            if not validate_smiles(smi):
                result["error"] = "无效的SMILES格式"
                results.append(result)
                continue
            
            # 提取特征
            mol = Chem.MolFromSmiles(smi)
            feats, feat_names = feature_extractor(mol)
            
            # 特征格式处理
            if isinstance(feats, (pd.Series, pd.DataFrame)):
                if feats.empty:
                    result["error"] = "特征提取为空"
                    results.append(result)
                    continue
                feats = feats.to_numpy()
            
            if feats is None or len(feats) == 0:
                result["error"] = "特征提取失败"
                results.append(result)
                continue
            
            # 特征对齐
            aligned_feats = align_features_static(feats, feat_names, feature_names_train)
            
            # 模型预测
            prob = model.predict_proba([aligned_feats])[:, 1][0]
            result["pre_probability"] = float(prob)
            result["pre_category"] = "positive" if prob >= threshold else "negative"
            result["threshold_used"] = threshold
            
        except Exception as e:
            result["error"] = f"处理失败: {str(e)}"
        
        results.append(result)
    
    return jsonify({
        "model_used": model_name,
        "results": results,
        "count": len(results),
        "success_count": sum(1 for r in results if "error" not in r)
    })

# 多模型预测接口
@app.route('/api/predict/all', methods=['POST'])
def predict_all_models():
    """使用所有模型进行预测并汇总结果"""
    data = request.json
    smiles_list = data.get('smiles_list', [])
    
    if not isinstance(smiles_list, list) or len(smiles_list) == 0:
        return jsonify({"error": "请提供有效的SMILES列表"}), 400
    
    results = []
    for smi in smiles_list:
        smi_result = {"smiles": smi, "models": {}}
        try:
            if not validate_smiles(smi):
                smi_result["error"] = "无效的SMILES格式"
                results.append(smi_result)
                continue
            
            # 收集所有模型的预测结果
            model_probs = []
            for model_name, model_data in models.items():
                try:
                    # 特征提取与预测
                    mol = Chem.MolFromSmiles(smi)
                    feats, feat_names = model_data["feature_extractor"](mol)
                    aligned_feats = align_features_static(feats, feat_names, model_data["feature_names"])
                    prob = model_data["model"].predict_proba([aligned_feats])[:, 1][0]
                    
                    model_probs.append(prob)
                    smi_result["models"][model_name] = {
                        "probability": float(prob),
                        "threshold": model_data["default_threshold"],
                        "category": "positive" if prob >= model_data["default_threshold"] else "negative"
                    }
                except Exception as e:
                    smi_result["models"][model_name] = {"error": str(e)}
            
            # 计算汇总结果
            if model_probs:
                avg_prob = np.mean(model_probs)
                model2_prob = smi_result["models"].get("Model 2", {}).get("probability", 0)
                smi_result["combined"] = {
                    "average_probability": float(avg_prob),
                    "model2_probability": float(model2_prob),
                    "final_category": "positive" if avg_prob >= 0.94 or model2_prob >= 0.93 else "negative",
                    "threshold_used": "avg >=0.94 or model2 >=0.93"
                }
        
        except Exception as e:
            smi_result["error"] = f"处理失败: {str(e)}"
        
        results.append(smi_result)
    
    return jsonify({
        "results": results,
        "count": len(results),
        "model_count": len(models)
    })

# 启动时加载模型并运行服务
if __name__ == '__main__':
    # 确保models目录存在
    if not os.path.exists('models/'):
        os.makedirs('models/')
    
    # 加载模型
    load_all_models()
    
    # 启动服务
    app.run(host='0.0.0.0', port=5000, debug=True)
