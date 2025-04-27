import React, { useState, useEffect } from 'react';
import { 
  Box, Typography, Paper, Button, Grid, CircularProgress, Alert,
  Card, CardContent, CardActions, CardMedia, Chip, Dialog, DialogTitle,
  DialogContent, DialogActions, IconButton, Tabs, Tab, Tooltip, Divider
} from '@mui/material';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import TextSnippetIcon from '@mui/icons-material/TextSnippet';
import VisibilityIcon from '@mui/icons-material/Visibility';
import RefreshIcon from '@mui/icons-material/Refresh';
import FormatSizeIcon from '@mui/icons-material/FormatSize';
import DateRangeIcon from '@mui/icons-material/DateRange';
import axios from 'axios';

// TabPanel component for filter tabs
function TabPanel(props) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 2 }}>{children}</Box>}
    </div>
  );
}

const DocumentsTab = ({ language }) => {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [textContent, setTextContent] = useState('');
  const [viewDialogOpen, setViewDialogOpen] = useState(false);
  const [currentDocument, setCurrentDocument] = useState(null);
  const [filterTab, setFilterTab] = useState(0); // 0: All, 1: PDF, 2: Text
  
  // Text translations
  const texts = {
    title: language === 'it' ? 'Documenti' : 'Documents',
    view: language === 'it' ? 'Visualizza' : 'View',
    close: language === 'it' ? 'Chiudi' : 'Close',
    refresh: language === 'it' ? 'Aggiorna' : 'Refresh',
    all: language === 'it' ? 'Tutti' : 'All',
    pdf: 'PDF',
    text: language === 'it' ? 'Testo' : 'Text',
    noDocs: language === 'it' ? 'Nessun documento disponibile' : 'No documents available',
    hasPdf: language === 'it' ? 'Ha PDF' : 'Has PDF',
    loading: language === 'it' ? 'Caricamento...' : 'Loading...',
    errorFetching: language === 'it' ? 'Errore nel recuperare i documenti' : 'Error fetching documents',
    errorViewing: language === 'it' ? 'Errore nel visualizzare il contenuto' : 'Error viewing content'
  };

  // Load documents on component mount
  useEffect(() => {
    fetchDocuments();
  }, []);

  // Format file size
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };
  
  // Format date
  const formatDate = (timestamp) => {
    return new Date(timestamp * 1000).toLocaleDateString();
  };
  
  // Fetch documents from API
  const fetchDocuments = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get('http://localhost:8000/list-all-documents');
      setDocuments(response.data.documents);
    } catch (err) {
      console.error('Error fetching documents:', err);
      setError(texts.errorFetching);
    } finally {
      setLoading(false);
    }
  };
  
  // View text content
  const viewTextContent = async (filename) => {
    try {
      const response = await axios.get(`http://localhost:8000/view-text-content/${filename}`);
      setTextContent(response.data.content);
      setViewDialogOpen(true);
    } catch (err) {
      console.error('Error viewing text content:', err);
      setError(texts.errorViewing);
    }
  };

  // Handle filter tab change
  const handleFilterChange = (event, newValue) => {
    setFilterTab(newValue);
  };

  // Set current document and open dialog
  const handleViewDocument = (doc) => {
    setCurrentDocument(doc);
    if (doc.type === 'text') {
      viewTextContent(doc.name);
    }
  };

  // Close dialog
  const handleCloseDialog = () => {
    setViewDialogOpen(false);
    setTextContent('');
    setCurrentDocument(null);
  };
  
  // Filter documents based on selected tab
  const filteredDocuments = documents.filter(doc => {
    if (filterTab === 0) return true; // All documents
    if (filterTab === 1) return doc.type === 'pdf'; // PDF only
    if (filterTab === 2) return doc.type === 'text'; // Text only
    return true;
  });

  // Helper function to render the document grid
  const renderDocumentGrid = (docs) => {
    if (loading) {
      return (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
          <CircularProgress />
        </Box>
      );
    }
    
    if (docs.length === 0) {
      return (
        <Typography variant="body1" sx={{ my: 4, textAlign: 'center' }}>
          {texts.noDocs}
        </Typography>
      );
    }
    
    return (
      <Grid container spacing={3} sx={{ mt: 1 }}>
        {docs.map((doc, index) => (
          <Grid item xs={12} sm={6} md={4} key={index}>
            <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <CardMedia
                sx={{ 
                  height: 140, 
                  bgcolor: doc.type === 'pdf' ? 'error.light' : 'primary.light',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}
              >
                {doc.type === 'pdf' ? (
                  <PictureAsPdfIcon sx={{ fontSize: 80, color: 'white' }} />
                ) : (
                  <TextSnippetIcon sx={{ fontSize: 80, color: 'white' }} />
                )}
              </CardMedia>
              <CardContent sx={{ flexGrow: 1 }}>
                <Typography gutterBottom variant="h6" component="div" noWrap>
                  {doc.name}
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 1 }}>
                  <Chip 
                    size="small" 
                    label={doc.type.toUpperCase()} 
                    color={doc.type === 'pdf' ? 'error' : 'primary'} 
                    variant="outlined"
                  />
                  {doc.type === 'text' && doc.has_pdf && (
                    <Chip size="small" label={texts.hasPdf} color="success" variant="outlined" />
                  )}
                </Box>
                <Divider sx={{ mb: 1 }} />
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <FormatSizeIcon fontSize="small" sx={{ mr: 1, color: 'text.secondary' }} />
                  <Typography variant="body2" color="text.secondary">
                    {formatFileSize(doc.size)}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <DateRangeIcon fontSize="small" sx={{ mr: 1, color: 'text.secondary' }} />
                  <Typography variant="body2" color="text.secondary">
                    {formatDate(doc.modified)}
                  </Typography>
                </Box>
              </CardContent>
              <CardActions>
                <Button 
                  size="small" 
                  startIcon={<VisibilityIcon />}
                  onClick={() => handleViewDocument(doc)}
                  disabled={doc.type === 'pdf'} // Can only view text files directly
                >
                  {texts.view}
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>
    );
  };
  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h4" component="h1">
          {texts.title}
        </Typography>
        
        <Tooltip title={texts.refresh}>
          <IconButton onClick={fetchDocuments} color="primary">
            <RefreshIcon />
          </IconButton>
        </Tooltip>
      </Box>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      
      <Paper elevation={3} sx={{ p: 3 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={filterTab} onChange={handleFilterChange} aria-label="document type tabs">
            <Tab label={texts.all} />
            <Tab label={texts.pdf} icon={<PictureAsPdfIcon />} iconPosition="start" />
            <Tab label={texts.text} icon={<TextSnippetIcon />} iconPosition="start" />
          </Tabs>
        </Box>
        
        <TabPanel value={filterTab} index={0}>
          {renderDocumentGrid(filteredDocuments)}
        </TabPanel>
        <TabPanel value={filterTab} index={1}>
          {renderDocumentGrid(filteredDocuments)}
        </TabPanel>
        <TabPanel value={filterTab} index={2}>
          {renderDocumentGrid(filteredDocuments)}
        </TabPanel>
      </Paper>
      
      {/* View Text Content Dialog */}
      <Dialog 
        open={viewDialogOpen} 
        onClose={handleCloseDialog}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {currentDocument?.name || ''}
        </DialogTitle>
        <DialogContent dividers>
          <Box sx={{ 
            whiteSpace: 'pre-wrap', 
            fontFamily: 'monospace', 
            maxHeight: '60vh', 
            overflow: 'auto',
            bgcolor: 'action.hover',
            p: 2,
            borderRadius: 1
          }}>
            {textContent}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>
            {texts.close}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default DocumentsTab;
