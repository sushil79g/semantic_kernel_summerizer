import semantic_kernel as sk
from semantic_kernel.connectors.ai.ollama import OllamaTextCompletion
import requests
import asyncio
from typing import Dict, Any

class OllamaResponseFunction(sk.functions.KernelFunction):
    def __init__(self, model: str, url: str):
        """Initialize with Ollama model and API URL"""
        super().__init__(name="OllamaResponse", description="Send requests to Ollama API")
        self.model = model
        self.url = url
    
    async def invoke(self, kernel: sk.Kernel, arguments: Dict[str, Any]) -> str:
        prompt = arguments.get("input", "")
        
        header = {"Content-Type": "application/json"}
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }

        output = requests.post(self.url, headers=header, json=data)
        
        if output.status_code == 200:
            return output.json().get("message", {}).get("content", "")
        else:
            return f"Error {output.status_code}: {output.text}"

async def main():
    kernel = sk.Kernel()
    
    ollama_url = "http://localhost:11434/api/chat"
    ollama_model = "qwq:latest"
    ollama_function = OllamaResponseFunction(model=ollama_model, url=ollama_url)
    
    ollama_plugin = kernel.create_plugin(name="OllamaPlugin", functions=[ollama_function])

    summarize_prompt = """
    Summarize the following text in 3-5 sentences:

    {{$input}}
    """
    
    summarize_function = kernel.create_function_from_prompt(
        function_name="SummarizeText",
        plugin_name="Utils",
        prompt=summarize_prompt
    )
    
    text_to_summarize = """
    The home page displays a list of templates to choose from, organized into four possible groups:
    
    Group   Description  Notes
    Installed Templates installed to your local computer. When an online template is used, its repository is automatically cloned to a subfolder of ~/.cookiecutters. You can remove an installed template from your system by selecting Delete on the Cookiecutter Explorer toolbar.
    Recommended Templates loaded from the recommended feed. Microsoft curates the default feed. You can customize the feed by following the steps in Set Cookiecutter options.
    GitHub GitHub search results for the "cookiecutter" keyword. The list of git repositories are returned in paginated form. When the list of results exceeds the current view, you can select the Load More option to show the next set of paginated results in the list.
    Custom Any custom templates defined through Cookiecutter Explorer. When a custom template location is entered in the Cookiecutter Explorer search box, the location appears in this group. You can define a custom template by entering the full path to the git repository, or the full path to a folder on your local disk.
    """
    summary_result = await kernel.invoke(
        summarize_function,
        input=text_to_summarize
    )
    
    final_result = await kernel.invoke(
        ollama_function,
        input=str(summary_result)
    )
    
    return str(final_result)

if __name__ == "__main__":
    output = asyncio.run(main())
    print(output)