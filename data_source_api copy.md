# API Access Example & Troubleshooting Guide

This document provides a step-by-step explanation of the complete cURL command example for accessing financial data via the OpenBDAP API, along with detailed usage instructions and troubleshooting tips.

---

## 1. Complete cURL Command Example

Below is the complete cURL command formatted using a Markdown code block. This command is used to retrieve a dataset from the API:

```bash
# Replace YOUR_API_TOKEN with your actual API token
# Replace {dataset_id} with the specific dataset ID you want to access
# Replace {param1}, {param2}, etc. with the required parameters

curl -X GET \
     -H 'Authorization: Token YOUR_API_TOKEN' \
     -d '{param1}=value1&{param2}=value2' \
     https://bdap-opendata.rgs.mef.gov.it/SpodCkanApi/api/1/rest/dataset/{dataset_id}
```

---

## 2. Breakdown of the Command Components

### a. Endpoint

- **URL**: `https://bdap-opendata.rgs.mef.gov.it/SpodCkanApi/api/1/rest/dataset/{dataset_id}`
- **Action**: This endpoint is used to access a specific dataset. 
- **Instructions**: Replace `{dataset_id}` with the unique identifier of the dataset you intend to retrieve.

### b. Authentication

- **Header**: `-H 'Authorization: Token YOUR_API_TOKEN'`
- **Action**: The `Authorization` header allows you to authenticate with the API using your API token.
- **Instructions**: Replace `YOUR_API_TOKEN` with your actual API token provided by the API service.

### c. Parameters

- **Option**: `-d '{param1}=value1&{param2}=value2'`
- **Action**: The `-d` flag is used to send parameters in the request body.
- **Instructions**: Replace `{param1}`, `{param2}`, etc., with the required parameter names, and `value1`, `value2`, etc., with their corresponding values. Multiple parameters are separated by an ampersand (`&`).

---

## 3. Step-by-Step Usage Instructions

1. **Obtain Your API Token**: Ensure you have received your API token from your API provider.
2. **Identify the Dataset ID**: Locate the dataset you wish to access and note its unique identifier.
3. **Customize the cURL Command**:
   - Replace `YOUR_API_TOKEN` with your actual API token.
   - Replace `{dataset_id}` in the URL with the dataset's identifier.
   - Replace placeholder parameters such as `{param1}` and `{param2}` with actual parameter names and set their corresponding values.
4. **Run the Command**: Execute the command in your terminal or command-line interface.

---

## 4. Troubleshooting Tips & Common Error Messages

### a. Common Issues

- **Invalid API Token**: 
  - **Issue**: The token provided is incorrect or has expired.
  - **Solution**: Verify your API token and replace it in the command.

- **Incorrect Dataset ID**:
  - **Issue**: The dataset ID provided does not exist or is invalid.
  - **Solution**: Double-check the dataset ID and ensure it matches the one provided in the API documentation.

- **Improper Parameter Formatting**:
  - **Issue**: Parameters not correctly formatted or missing required values.
  - **Solution**: Ensure parameters follow the `{param_name}=value` format and multiple parameters are joined with `&`.

- **Network Issues**:
  - **Issue**: Connectivity problems may cause the request to fail.
  - **Solution**: Check your internet connection and try the command again.

### b. Example Error Message

- **Error**: `401 Unauthorized`
  - **Meaning**: The API token is invalid or missing.
  - **Troubleshooting**: Confirm that you have inserted a valid API token in the command.

- **Error**: `404 Not Found`
  - **Meaning**: The provided endpoint (e.g., dataset ID) does not exist.
  - **Troubleshooting**: Validate the dataset ID and the URL.

---

## 5. Summary of Usage Guidelines for Accessing Financial Data via the API

- **Authentication**: Always use your valid API token in the `Authorization` header.
- **Endpoint & Dataset ID**: Ensure the endpoint is correctly targeted with a valid dataset ID.
- **Parameters**: Review and correctly include all required parameters using the correct formatting.
- **Troubleshooting**: Refer to the common error messages and troubleshooting tips if you encounter issues.

For further information, please refer to the [API Documentation](https://bdap-opendata.rgs.mef.gov.it/SpodCkanApi/api/1/rest/dataset). (Link from Agent 4 output)

---

This guide should help you successfully access and retrieve financial data using the provided API endpoint via cURL. Follow the steps precisely and use the troubleshooting tips if issues arise.
