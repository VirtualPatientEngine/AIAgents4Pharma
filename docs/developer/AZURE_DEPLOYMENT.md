# Azure OpenAI Deployment Guide

This guide provides instructions for configuring AIAgents4Pharma to work with Azure OpenAI services, which is particularly useful for enterprise environments that require custom endpoints or Azure-based deployments.

## Overview

AIAgents4Pharma supports three OpenAI deployment modes:
1. **Standard OpenAI** - Direct OpenAI API usage
2. **Custom OpenAI Endpoint** - OpenAI-compatible API with custom base URL
3. **Azure OpenAI** - Full Azure OpenAI service with Azure AD authentication

## Prerequisites

### For Azure OpenAI
- Azure subscription with OpenAI service enabled
- Azure OpenAI resource created in Azure portal
- Model deployed in Azure OpenAI Studio
- Azure AD application registered for authentication

### Required Python Dependencies
The following packages are already included in the project requirements:
- `langchain-openai` (includes AzureChatOpenAI)
- `azure-identity` (for Azure AD authentication)

## Configuration Options

### Option 1: Custom OpenAI Endpoint

If you have an OpenAI-compatible endpoint (e.g., enterprise proxy), set:

```bash
export OPENAI_API_KEY=your_api_key
export OPENAI_BASE_URL=https://your-custom-endpoint.com/v1
```

Then select `OpenAI/gpt-4o-mini` models in the application dropdown.

### Option 2: Azure OpenAI with Azure AD Authentication

For full Azure OpenAI deployment with Azure AD authentication:

#### Step 1: Azure OpenAI Resource Setup

1. Create an Azure OpenAI resource in the [Azure portal](https://portal.azure.com)
2. Deploy your desired model (e.g., GPT-4o-mini) in [Azure OpenAI Studio](https://oai.azure.com)
3. Note down:
   - **Endpoint URL**: `https://your-resource.openai.azure.com/`
   - **Deployment name**: Name you gave your model deployment
   - **API Version**: Recommended `2024-02-01`

#### Step 2: Azure AD Application Registration

1. Go to [Azure AD App Registrations](https://portal.azure.com/#view/Microsoft_AAD_RegisteredApps)
2. Create a new application registration
3. Add required API permissions for Azure OpenAI:
   - Go to "API permissions" → "Add a permission"
   - Select "Azure Service Management" → "Delegated permissions"
   - Add "user_impersonation" permission
4. Create a client secret in "Certificates & secrets"
5. Note down:
   - **Application (client) ID**
   - **Directory (tenant) ID**
   - **Client secret value**

#### Step 3: Environment Variables

Set the following environment variables:

```bash
# Azure OpenAI Configuration
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export AZURE_OPENAI_DEPLOYMENT=your-deployment-name
export AZURE_OPENAI_API_VERSION=2024-02-01
export AZURE_OPENAI_MODEL_NAME=gpt-4o-mini          # Optional: for analytics
export AZURE_OPENAI_MODEL_VERSION=1.0               # Optional: model version

# Azure AD Authentication
export AZURE_CLIENT_ID=your-application-client-id
export AZURE_TENANT_ID=your-directory-tenant-id
export AZURE_CLIENT_SECRET=your-client-secret

# Other required APIs (unchanged)
export NVIDIA_API_KEY=your_nvidia_key
# ... other environment variables
```

#### Step 4: Model Selection

In the application interface, select `Azure/gpt-4o-mini` from the LLM dropdown to use Azure OpenAI.

## Supported Models

Currently supported Azure OpenAI models:
- `Azure/gpt-4o-mini` - Maps to your Azure deployment

You can easily add more models by:
1. Adding them to the `azure_openai_llms` list in the Hydra config
2. Adding the mapping in `streamlit_utils.py`

## Troubleshooting

### Common Issues

**Error: "Azure OpenAI requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT"**
- Ensure both environment variables are set correctly
- Verify the endpoint URL format: `https://your-resource.openai.azure.com/`

**Error: "Failed to create Azure token provider"**
- Check Azure AD application permissions
- Verify client ID, tenant ID, and client secret are correct
- Ensure the application has proper permissions for Azure OpenAI

**Error: "Authentication failed"**
- Verify your Azure AD application has the required API permissions
- Check if the client secret has expired
- Ensure the service principal has access to the Azure OpenAI resource

### Testing Your Configuration

You can test Azure AD authentication manually:

```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

try:
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential,
        "https://cognitiveservices.azure.com/.default"
    )
    print("Azure authentication successful!")
except Exception as e:
    print(f"Authentication failed: {e}")
```

## Security Best Practices

1. **Use environment variables** for all sensitive information
2. **Rotate client secrets** regularly in Azure AD
3. **Use managed identities** when running on Azure resources
4. **Limit API permissions** to minimum required scope
5. **Monitor usage** in Azure OpenAI Studio

## Enterprise Deployment

For production deployments:

1. **Use Azure Key Vault** for storing secrets
2. **Configure network access rules** in Azure OpenAI
3. **Set up monitoring and logging** in Azure Monitor
4. **Use managed identities** instead of client secrets when possible
5. **Implement proper RBAC** for Azure OpenAI resources

## Migration from Standard OpenAI

To migrate from standard OpenAI to Azure OpenAI:

1. Set up Azure OpenAI resource and deployment
2. Configure environment variables as shown above
3. Change model selection from `OpenAI/gpt-4o-mini` to `Azure/gpt-4o-mini`
4. Test functionality with your use cases
5. Monitor costs and usage in Azure portal

## Cost Optimization

- Monitor token usage in Azure OpenAI Studio
- Set up billing alerts in Azure
- Consider using different model sizes based on use case complexity
- Implement rate limiting if needed

## Support

For Azure-specific issues:
- [Azure OpenAI Documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/openai/)
- [Azure Support](https://azure.microsoft.com/en-us/support/)

For AIAgents4Pharma issues:
- [GitHub Issues](https://github.com/VirtualPatientEngine/AIAgents4Pharma/issues)
- [Documentation](https://virtualpatientengine.github.io/AIAgents4Pharma/)
