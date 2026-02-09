from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseProvider(ABC):

    @abstractmethod
    def chatCompletion(self, prompt: str, save_messages: bool = False) -> str:
        """Get a chat completion from the provider."""
        pass

    @abstractmethod
    async def asyncChatCompletion(self, prompt: str, save_messages: bool = False) -> Any:
        """Get a chat completion asynchronously from the provider."""
        pass

    @abstractmethod
    def getProviderName(self) -> str:
        """Get the name of the provider."""
        pass

    @abstractmethod
    def getModels(self) -> List[str]:
        """Get the list of models available for the provider."""
        pass

    def setSystemPrompt(self, sys_prompt: str) -> Any:
        """Set the system prompt for the provider."""
        self.system_prompt = sys_prompt
        return self

    def setModel(self, model: str, validate: bool = False) -> Any:
        """Set the model for the provider."""
        if validate and model not in self.models:
            raise ValueError(f"Model {model} is not available for the provider.")
        
        self.model = model
        return self

    def setTemperature(self, temperature: float) -> Any:
        """Set the temperature for the provider."""
        if temperature < 0 or temperature > 1:
            raise ValueError("Temperature must be between 0 and 1.")
        
        self.temperature = temperature
        return self
    
    def setMaxTokens(self, max_tokens: int) -> Any:
        """Set the maximum tokens for the provider."""
        self.max_tokens = max_tokens
        return self

    def setTopP(self, top_p: float) -> Any:
        """Set the top-p for the provider."""
        if top_p < 0 or top_p > 1:
            raise ValueError("Top-P must be between 0 and 1.")
        
        self.top_p = top_p
        return self

    def setBaseUrl(self, url: str) -> Any:
        """Set the base URL for the provider."""
        self.base_url = url
        return self

    def setAPIKey(self, api_key: str) -> Any:
        """Set the API key for the provider."""
        self.api_key = api_key
        return self
    
    def getModelName(self) -> str:
        """Get the selected model for the provider."""
        return self.model_name

    def getConfig(self) -> Dict[str, str]:
        """Get the configuration for the provider."""
        return self.config
    
    def getBaseUrl(self) -> str:
        """Get the base URL for the provider."""
        return self.base_url
    
    def getMessages(self, user_message: str) -> List[dict]:
        """Build messages list for the API call."""
        if not self.messages:
            self.messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ]
        else:
            self.messages.append({"role": "user", "content": user_message})

        return self.messages
    
    def addAssistantMessage(self, assistant_message: str):
        """Add an assistant message to the conversation history."""
        self.messages.append({"role": "assistant", "content": assistant_message})

    def grounded_search(self, body: str):
        """Gets response from the grounding model completion."""
        pass