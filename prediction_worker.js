self.onmessage = function(e) {
    // 打印接收到的消息，用于调试
    console.log('Worker接收到消息:', e.data);
    
    const { model, threshold, smilesList } = e.data;
    
    // 验证输入数据
    if (!model || threshold === undefined || !smilesList || !Array.isArray(smilesList)) {
        const errorMsg = '输入数据不完整或格式错误';
        console.error(errorMsg, '接收的数据:', e.data);
        self.postMessage({
            type: 'error',
            message: errorMsg
        });
        return;
    }
    
    const total = smilesList.length;
    const chunkSize = 10;
    let completed = 0;
    const allResults = [];

    // 分块处理函数
    async function processChunks() {
        try {
            console.log(`开始处理，模型: ${model}, 阈值: ${threshold}, 总数量: ${total}`);
            
            for (let i = 0; i < smilesList.length; i += chunkSize) {
                const chunk = smilesList.slice(i, i + chunkSize);
                console.log(`处理分块 ${i/chunkSize + 1}: 包含 ${chunk.length} 个分子`);
                
                try {
                    const url = model === "Model all" ? '/api/predict/all' : '/api/predict';
                    console.log(`发送请求到: ${url}`);
                    
                    const response = await fetch(url, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            model: model,
                            threshold: threshold,
                            smiles_list: chunk
                        })
                    });

                    console.log(`服务器响应状态: ${response.status}`);
                    
                    // 读取原始响应内容用于调试
                    const responseText = await response.text();
                    console.log(`服务器响应内容: ${responseText.substring(0, 500)}${responseText.length > 500 ? '...' : ''}`);
                    
                    if (!response.ok) {
                        throw new Error(`服务器返回错误状态: ${response.status}, 内容: ${responseText.substring(0, 200)}`);
                    }

                    // 尝试解析JSON
                    let data;
                    try {
                        data = JSON.parse(responseText);
                    } catch (parseError) {
                        throw new Error(`解析服务器响应失败: ${parseError.message}, 原始内容: ${responseText.substring(0, 200)}`);
                    }
                    
                    if (!data.results || !Array.isArray(data.results)) {
                        throw new Error('服务器返回结果格式不正确，缺少results数组');
                    }

                    allResults.push(...data.results);
                    completed += chunk.length;
                    
                    // 发送进度更新
                    const progress = Math.floor((completed / total) * 100);
                    const progressMsg = `已完成 ${completed}/${total} 个分子预测`;
                    console.log(progressMsg);
                    
                    self.postMessage({
                        type: 'progress',
                        progress: 20 + Math.floor(progress * 0.8),
                        message: progressMsg
                    });

                } catch (chunkError) {
                    const errorMsg = `处理分块时出错: ${chunkError.message}`;
                    console.error(errorMsg, chunkError);
                    self.postMessage({
                        type: 'error',
                        message: errorMsg
                    });
                    return;
                }
            }

            // 所有分块处理完成
            console.log(`所有处理完成，共处理 ${completed} 个分子`);
            self.postMessage({
                type: 'result',
                results: allResults
            });
            
        } catch (error) {
            const errorMsg = `处理过程中出错: ${error.message}`;
            console.error(errorMsg, error);
            self.postMessage({
                type: 'error',
                message: errorMsg
            });
        }
    }

    // 开始处理
    processChunks();
};
