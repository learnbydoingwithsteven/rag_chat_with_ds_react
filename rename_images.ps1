# PowerShell script to rename image files to be GitHub-compatible
$assetPath = "assets"

# Create a mapping for each file
$fileMapping = @(
    @{
        "OldName" = "1. load reference documents.png"
        "NewName" = "1_load_reference_documents.png"
    },
    @{
        "OldName" = "2. check reference documents.png"
        "NewName" = "2_check_reference_documents.png"
    },
    @{
        "OldName" = "3. verify reference documents chunk embedding database.png"
        "NewName" = "3_verify_reference_documents_chunk_embedding_database.png"
    },
    @{
        "OldName" = "4. verify reference documents chunk embedding database scatter plots.png"
        "NewName" = "4_verify_reference_documents_chunk_embedding_database_scatter_plots.png"
    },
    @{
        "OldName" = "5. verify reference documents chunk embedding database scatter plots.png"
        "NewName" = "5_verify_reference_documents_chunk_embedding_database_scatter_plots.png"
    },
    @{
        "OldName" = "6. check available documents as reference, pdf and txt.png"
        "NewName" = "6_check_available_documents_as_reference_pdf_and_txt.png"
    },
    @{
        "OldName" = "7. rag chat interface with groq and ollama options.png"
        "NewName" = "7_rag_chat_interface_with_groq_and_ollama_options.png"
    },
    @{
        "OldName" = "8. public database api communication(under development).png"
        "NewName" = "8_public_database_api_communication_under_development.png"
    },
    @{
        "OldName" = "9. additional options to use microsoft formulator tool to add additional data visualization(gemini or openai api key, with option using ollama).png"
        "NewName" = "9_additional_options_to_use_microsoft_formulator_tool_to_add_additional_data_visualization_api_key_options.png"
    }
)

# Perform the renaming
foreach ($file in $fileMapping) {
    $oldPath = Join-Path -Path $assetPath -ChildPath $file.OldName
    $newPath = Join-Path -Path $assetPath -ChildPath $file.NewName
    
    if (Test-Path $oldPath) {
        Copy-Item -Path $oldPath -Destination $newPath -Force
        Write-Host "Copied: $($file.OldName) -> $($file.NewName)"
    } else {
        Write-Host "File not found: $($file.OldName)"
    }
}

Write-Host "Image renaming complete!"
