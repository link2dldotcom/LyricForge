document.getElementById('promptForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    const resultDiv = document.getElementById('result');
    const errorDiv = document.getElementById('error');
    const submitBtn = document.querySelector('.submit-btn');

    // 重置显示
    resultDiv.textContent = '生成中...';
    errorDiv.style.display = 'none';
    submitBtn.disabled = true;
    submitBtn.textContent = '生成中...';

    try {
        // 收集表单数据
        const formData = {
            mood: document.getElementById('mood').value,
            style: document.getElementById('style').value,
            language: document.getElementById('language').value,
            idea: document.getElementById('idea').value,
            // 收集选中的工具
            tools: Array.from(document.querySelectorAll('input[name="tools"]:checked'))
                .map(checkbox => checkbox.value)
        };

        // 发送请求到后端API
        const response = await fetch('/generate-suno-prompt', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });

        // 在获取响应后添加调试输出
        const data = await response.json();
        console.log("API返回数据:", data);  // 打开浏览器控制台查看此输出

        if (response.ok) {
            // 尝试多种可能的字段名，适应不同的后端返回格式
            const result = data.prompt || data.result || data.response || JSON.stringify(data);
            // 将\n替换为<br>标签并使用innerHTML显示
            resultDiv.innerHTML = result.replace(/\n/g, '<br>');
        } else {
            throw new Error(data.detail || data.error || '生成失败，请重试');
        }
    } catch (error) {
        errorDiv.textContent = error.message;
        errorDiv.style.display = 'block';
        resultDiv.textContent = '';
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = '生成提示词';
    }
});