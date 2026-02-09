# import all the llms in this module
from .base_provider import BaseProvider

from typing import List, Dict, Any

import importlib
import pkgutil
from pathlib import Path

class ProviderFactory:
    '''
    This is a factory class that is used to create instances of a provider.
    '''
    def __init__(self):
        '''
        Initializes the factory class by loading all the provider classes and instantiating them.
        '''
        self.top_module = ''
        self.provider_classes : Dict[str, BaseProvider] = self._load_provider_classes()
        self.all_providers : List[Dict[str, Any]] = self._instantiate_providers()
        

    def _load_provider_classes(self) -> Dict[str, BaseProvider]:
        '''
        Load all provider classes from the providers folder.
        '''
        # Use __file__ to get the directory of the current script
        factory_dir = Path(__file__).parent
        modules = self._find_modules_in_folder(factory_dir)

        # Collect all classes that inherit from BaseProvider
        provider_classes = {}
        # Iterate through all modules
        for module_name in modules:
            classes = self._load_classes_from_module(module_name)
            for cls in classes:
                # Store the class in the provider_classes dictionary
                provider_classes[cls.__name__] = cls
        return provider_classes

    def _find_modules_in_folder(self, folder_path):
        '''
        Find all modules in the folder.
        '''
        modules = []
        for _, module_name, _ in pkgutil.iter_modules([str(folder_path)]):
            # Use the full module path (e.g., "folder.provider_a")
            modules.append(f"{folder_path.name}.{module_name}")
        return modules

    def _load_classes_from_module(self, module_name) -> List[BaseProvider]:
        '''
        Load all classes from the module that inherit from BaseProvider.
        '''
        
        module_path = self.top_module + module_name
        module = importlib.import_module(module_path)
        classes = []
        for name, obj in module.__dict__.items():
            # Check if the object is a class and inherits from BaseProvider
            if isinstance(obj, type) and issubclass(obj, BaseProvider) and obj is not BaseProvider:
                classes.append(obj)
        return classes
    
    def _instantiate_providers(self) -> List[Dict[str, Any]]:
        '''
        Instantiate all provider classes to extract the provider name, models, and config.
        Returns a list of dictionaries containing the provider name, class, models, and config.
        '''
        all_providers = []
        # Iterate through all provider classes
        for cls_name, cls in self.provider_classes.items():
            # Instantiate the provider
            provider_instance = cls(('fake_key'), **{'model': 'fake_model'})
            # store a dict of values to the all_providers list
            all_providers.append({
                    'provider_name': provider_instance.getProviderName(), 
                    'cls': cls,
                    'models': provider_instance.getModels(),
                    'config': provider_instance.getConfig(),
                    'base_url': provider_instance.getBaseUrl()
                    })
        return all_providers
    

    def get_provider_instance(self, **kwargs) -> BaseProvider:
        '''
        Get the provider instance by provider_name or base_url.
        Args:
            provider_name: The name of the provider.
            base_url: The base URL of the provider.
            **kwargs: Dictionary arguments to pass to the provider class.
        Returns:
            The instance of the provider class.
        '''
        provider_name = kwargs.get("provider_name", None)
        base_url = kwargs.get("base_url", None)

        for provider in self.all_providers:
            if base_url is not None and provider["base_url"] == base_url:
                return provider["cls"](**kwargs)
            if provider_name is not None and provider["provider_name"] == provider_name.lower():
                return provider["cls"](**kwargs)
        raise ValueError("Provider not found")

    def get_all_providers_names(self) -> List[str]:
        '''
        Get all the provider names.
        Returns:
            A list of provider names.
        '''
        return [provider['provider_name'] for provider in self.all_providers] 
        
    def get_config_for_provider(self, provider_name) -> dict:
        '''
        Get the config for a given provider.
        Args:
            provider_name: The name of the provider.
            Returns:
                The config for the provider.
        '''
        for provider in self.all_providers:
            if provider['provider_name'] == provider_name:
                return provider['config']
        raise ValueError(f"Provider {provider_name} not found")
    
    def get_all_models_for_provider(self, provider_name) -> List[str]:
        '''
        Get all models for a given provider.
        Args:
            provider_name: The name of the provider.
        Returns:
            A list of models for the provider.
        '''
        for provider in self.all_providers:
            if provider['provider_name'] == provider_name:
                return provider['models']
        raise ValueError(f"Provider {provider_name} not found")

    def get_all_providers(self) -> List[Dict[str, Any]]:
        '''
        Get all providers.
        Returns:
            A list of dictionaries containing the provider name, class, models, and config.
        '''
        return self.all_providers