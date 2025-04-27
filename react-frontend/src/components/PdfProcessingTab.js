import React, { useState, useEffect } from 'react';
import { 
  Box, Button, Typography, Paper, List, ListItem, 
  ListItemText, CircularProgress, Alert, Divider, Grid,
  Card, CardContent, Checkbox, ListItemIcon, IconButton
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';
import RefreshIcon from '@mui/icons-material/Refresh';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import axios from 'axios';

const PdfProcessingTab = ({ language }) => {
  const [pdfFiles, setPdfFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [processingResults, setProcessingResults] = useState(null);
  const [processingTaskId, setProcessingTaskId] = useState(null);
  const [error, setError] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedPdfs, setSelectedPdfs] = useState([]);
  const [directoryInfo, setDirectoryInfo] = useState(null);

  // UI text based on selected language
  const texts = {
    title: language === 'it' ? 'Elaborazione PDF' : 'PDF Processing',
    subtitle: language === 'it' 
      ? 'Gestisci e converti i file PDF in formato testo per l\'elaborazione' 
      : 'Manage and convert PDF files to text format for processing',
    scanDir: language === 'it' ? 'Scansiona Directory PDF' : 'Scan PDF Directory',
    scanningDir: language === 'it' ? 'Scansione in corso...' : 'Scanning directory...',
    currentPdfs: language === 'it' ? 'File PDF disponibili' : 'Available PDF files',
    upload: language === 'it' ? 'Carica PDF' : 'Upload PDF',
    chooseFile: language === 'it' ? 'Scegli file' : 'Choose file',
    processSelected: language === 'it' ? 'Elabora selezionati' : 'Process Selected',
    processAll: language === 'it' ? 'Elabora tutti i PDF' : 'Process all PDFs',
    processingStatus: language === 'it' ? 'Stato elaborazione' : 'Processing status',
    noFiles: language === 'it' ? 'Nessun file PDF trovato' : 'No PDF files found',
    uploadSuccess: language === 'it' ? 'File caricato con successo' : 'File uploaded successfully',
    processingStarted: language === 'it' ? 'Elaborazione avviata' : 'Processing started',
    checkingStatus: language === 'it' ? 'Controllo stato...' : 'Checking status...',
    completed: language === 'it' ? 'Completato' : 'Completed',
    error: language === 'it' ? 'Errore' : 'Error',
    running: language === 'it' ? 'In esecuzione' : 'Running',
    fileName: language === 'it' ? 'Nome file' : 'Filename',
    status: language === 'it' ? 'Stato' : 'Status',
    details: language === 'it' ? 'Dettagli' : 'Details',
    refreshList: language === 'it' ? 'Aggiorna lista' : 'Refresh list',
    selectAll: language === 'it' ? 'Seleziona tutti' : 'Select all',
    deselectAll: language === 'it' ? 'Deseleziona tutti' : 'Deselect all',
    selectedFiles: language === 'it' ? 'File selezionati' : 'Selected files',
    scanDescription: language === 'it' 
      ? 'Inizia controllando quali PDF sono disponibili nella directory' 
      : 'Start by checking which PDFs are available in the directory',
    directoryPath: language === 'it' ? 'Percorso directory' : 'Directory path',
    moveFiles: language === 'it' ? 'Sposta i tuoi file PDF nella directory' : 'Move your PDF files to the directory',
    directoryNote: language === 'it' 
      ? 'Nota: I file PDF devono essere nella directory appropriata per essere trovati' 
      : 'Note: PDF files must be in the appropriate directory to be found',
  };

  // Initial setup - don't auto-fetch files
  useEffect(() => {
    // Reset UI state when language changes
    setProcessingResults(null);
    setError(null);
  }, [language]);

  // Poll processing status if a task is running
  useEffect(() => {
    let intervalId;
    if (processingTaskId) {
      intervalId = setInterval(() => {
        checkProcessingStatus(processingTaskId);
      }, 2000);
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [processingTaskId]);

  // Scan directory for PDF files
  const scanPdfDirectory = async () => {
    setScanning(true);
    setError(null);
    setProcessingResults(null);
    setSelectedPdfs([]);
    
    try {
      const response = await axios.get('http://localhost:8000/list-pdf-files');
      // Handle files that might be objects with filename property or direct strings
      const files = response.data.files || [];
      setPdfFiles(files);
      setDirectoryInfo({
        path: response.data.directory,
        exists: response.data.exists,
        note: response.data.note
      });
      setScanning(false);
    } catch (err) {
      console.error('Error scanning PDF directory:', err);
      setError(texts.error + ': ' + (err.response?.data?.detail || err.message));
      setScanning(false);
    }
  };

  // Handle file selection for upload
  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };
  
  // Toggle selection of a PDF file
  const togglePdfSelection = (file) => {
    // Handle both string filenames and file objects
    const filename = typeof file === 'object' ? file.filename : file;
    
    if (selectedPdfs.some(f => (typeof f === 'object' ? f.filename : f) === filename)) {
      setSelectedPdfs(selectedPdfs.filter(f => (typeof f === 'object' ? f.filename : f) !== filename));
    } else {
      setSelectedPdfs([...selectedPdfs, file]);
    }
  };
  
  // Select or deselect all PDFs
  const toggleSelectAll = () => {
    // Compare length to determine if all files are selected
    if (selectedPdfs.length === pdfFiles.length) {
      // Deselect all
      setSelectedPdfs([]);
    } else {
      // Select all
      setSelectedPdfs([...pdfFiles]);
    }
  };

  // Upload a PDF file
  const handleUpload = async () => {
    if (!selectedFile) return;
    
    setUploading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
      const response = await axios.post('http://localhost:8000/upload-pdf', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      // Clear selected file and refresh list
      setSelectedFile(null);
      scanPdfDirectory();
      setUploading(false);
      
      // Show success alert briefly
      setProcessingResults({
        title: texts.uploadSuccess,
        results: [response.data]
      });
      
      // Clear file input
      const fileInput = document.getElementById('pdf-file-input');
      if (fileInput) fileInput.value = '';
      
    } catch (err) {
      console.error('Error uploading file:', err);
      setError(texts.error + ': ' + (err.response?.data?.detail || err.message));
      setUploading(false);
    }
  };

  // Process selected PDF files
  const handleProcessSelectedPdfs = async () => {
    if (selectedPdfs.length === 0) return;
    
    setProcessing(true);
    setProcessingResults(null);
    setError(null);
    
    try {
      // Extract filenames if files are objects
      const filesToProcess = selectedPdfs.map(file => 
        typeof file === 'object' ? file.filename : file
      );
      
      const response = await axios.post('http://localhost:8000/process-pdf-files', {
        files: filesToProcess
      });
      setProcessingTaskId(response.data.task_id);
      setProcessingResults({
        title: texts.processingStarted,
        status: 'running'
      });
    } catch (err) {
      console.error('Error processing PDFs:', err);
      setError(texts.error + ': ' + (err.response?.data?.detail || err.message));
      setProcessing(false);
    }
  };
  
  // Process all PDF files
  const handleProcessAllPdfs = async () => {
    setProcessing(true);
    setProcessingResults(null);
    setError(null);
    
    try {
      const response = await axios.post('http://localhost:8000/process-pdf-files');
      setProcessingTaskId(response.data.task_id);
      setProcessingResults({
        title: texts.processingStarted,
        status: 'running'
      });
    } catch (err) {
      console.error('Error processing PDFs:', err);
      setError(texts.error + ': ' + (err.response?.data?.detail || err.message));
      setProcessing(false);
    }
  };

  // Check processing status
  const checkProcessingStatus = async (taskId) => {
    try {
      const response = await axios.get(`http://localhost:8000/process-pdf-status/${taskId}`);
      
      if (response.data.status === 'completed') {
        setProcessing(false);
        setProcessingTaskId(null);
        setProcessingResults({
          title: texts.completed,
          results: response.data.results,
          status: 'completed'
        });
        // Refresh the file list
        scanPdfDirectory();
      } else if (response.data.status === 'error') {
        setProcessing(false);
        setProcessingTaskId(null);
        setError(texts.error + ': ' + response.data.error);
      } else {
        // Still running
        setProcessingResults({
          title: texts.checkingStatus,
          status: 'running'
        });
      }
    } catch (err) {
      console.error('Error checking processing status:', err);
      setError(texts.error + ': ' + (err.response?.data?.detail || err.message));
      setProcessing(false);
      setProcessingTaskId(null);
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        {texts.title}
      </Typography>
      
      <Typography variant="body1" paragraph>
        {texts.subtitle}
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      
      {/* Upload Section */}
      <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          {texts.upload}
        </Typography>
        
        <Grid container spacing={2} alignItems="center">
          <Grid item>
            <input
              accept="application/pdf"
              id="pdf-file-input"
              type="file"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
            <label htmlFor="pdf-file-input">
              <Button
                variant="contained"
                component="span"
                startIcon={<CloudUploadIcon />}
              >
                {texts.chooseFile}
              </Button>
            </label>
          </Grid>
          
          <Grid item>
            {selectedFile && (
              <Typography variant="body2">
                {selectedFile.name}
              </Typography>
            )}
          </Grid>
          
          <Grid item>
            <Button
              variant="contained"
              color="primary"
              onClick={handleUpload}
              disabled={!selectedFile || uploading}
            >
              {uploading ? <CircularProgress size={24} /> : texts.upload}
            </Button>
          </Grid>
        </Grid>
      </Paper>
      
      {/* Scan Directory Section */}
      <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          {texts.scanDir}
        </Typography>
        
        <Typography variant="body2" paragraph>
          {texts.scanDescription}
        </Typography>
        
        <Button
          variant="contained"
          color="primary"
          startIcon={<FolderOpenIcon />}
          onClick={scanPdfDirectory}
          disabled={scanning}
          sx={{ mb: 2 }}
        >
          {scanning ? <CircularProgress size={24} /> : texts.scanDir}
        </Button>
        
        {directoryInfo && (
          <Box sx={{ mt: 2, p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
            <Typography variant="subtitle2" gutterBottom>
              {texts.directoryPath}:
            </Typography>
            <Typography variant="body2" component="div" sx={{ wordBreak: 'break-all', mb: 1, fontFamily: 'monospace', bgcolor: 'action.hover', p: 1, borderRadius: 1 }}>
              {directoryInfo.path}
            </Typography>
            
            {pdfFiles.length === 0 && (
              <Alert severity="info" sx={{ mt: 2 }}>
                {texts.moveFiles}: <strong>{directoryInfo.path}</strong>
              </Alert>
            )}
            
            {directoryInfo.note && (
              <Alert severity="warning" sx={{ mt: 2 }}>
                {directoryInfo.note}
              </Alert>
            )}
          </Box>
        )}
      </Paper>
      
      {/* Process PDFs Section */}
      <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          {texts.processAll}
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
          <Button
            variant="contained"
            color="secondary"
            onClick={handleProcessSelectedPdfs}
            disabled={processing || selectedPdfs.length === 0}
          >
            {processing ? <CircularProgress size={24} /> : texts.processSelected} 
            {selectedPdfs.length > 0 && ` (${selectedPdfs.length})`}
          </Button>
          
          <Button
            variant="outlined"
            color="secondary"
            onClick={handleProcessAllPdfs}
            disabled={processing || pdfFiles.length === 0}
          >
            {processing ? <CircularProgress size={24} /> : texts.processAll}
          </Button>
        </Box>
        
        {processingResults && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle1">
              {processingResults.title}
            </Typography>
            
            {processingResults.status === 'running' && (
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <CircularProgress size={20} sx={{ mr: 1 }} />
                <Typography variant="body2">
                  {texts.running}
                </Typography>
              </Box>
            )}
            
            {processingResults.results && (
              <List>
                {processingResults.results.map((result, index) => (
                  <ListItem key={index} divider={index < processingResults.results.length - 1}>
                    <ListItemText
                      primary={result.file}
                      secondary={
                        <>
                          <Typography component="span" variant="body2" color={result.status === 'success' ? 'success.main' : 'error.main'}>
                            {result.status === 'success' ? texts.completed : texts.error}
                          </Typography>
                          {result.status === 'success' && (
                            <Typography component="span" variant="body2">
                              {` • ${result.text_file} • ${result.duration} • ${result.text_length} chars`}
                            </Typography>
                          )}
                          {result.status === 'error' && (
                            <Typography component="span" variant="body2">
                              {` • ${result.message}`}
                            </Typography>
                          )}
                        </>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            )}
          </Box>
        )}
      </Paper>
      
      {/* File List Section */}
      <Paper elevation={3} sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">
            {texts.currentPdfs} {pdfFiles.length > 0 && `(${pdfFiles.length})`}
          </Typography>
          <Box>
            <Button 
              variant="text" 
              onClick={toggleSelectAll} 
              size="small"
              disabled={pdfFiles.length === 0}
              sx={{ mr: 1 }}
            >
              {selectedPdfs.length === pdfFiles.length ? texts.deselectAll : texts.selectAll}
            </Button>
            <IconButton onClick={scanPdfDirectory} size="small" color="primary">
              <RefreshIcon />
            </IconButton>
          </Box>
        </Box>
        
        <Divider sx={{ mb: 2 }} />
        
        {pdfFiles.length === 0 ? (
          <Box sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              {scanning ? texts.scanningDir : texts.noFiles}
            </Typography>
            {!scanning && directoryInfo && (
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                {texts.directoryNote}
              </Typography>
            )}
          </Box>
        ) : (
          <List>
            {pdfFiles.map((file, index) => (
              <ListItem 
                key={index} 
                divider={index < pdfFiles.length - 1}
                dense
                button
                onClick={() => togglePdfSelection(file)}
              >
                <ListItemIcon>
                  <Checkbox
                    edge="start"
                    checked={selectedPdfs.some(f => 
                      (typeof f === 'object' && typeof file === 'object') 
                        ? f.filename === file.filename
                        : f === file
                    )}
                    tabIndex={-1}
                    disableRipple
                  />
                </ListItemIcon>
                <ListItemIcon>
                  <PictureAsPdfIcon color="error" fontSize="small" />
                </ListItemIcon>
                <ListItemText primary={typeof file === 'object' ? file.filename : file} />
              </ListItem>
            ))}
          </List>
        )}
      </Paper>
    </Box>
  );
};

export default PdfProcessingTab;
