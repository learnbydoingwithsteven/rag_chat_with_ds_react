import React, { useState, useEffect } from 'react';
import { 
  AppBar, 
  Tabs, 
  Tab, 
  Box, 
  Container, 
  ThemeProvider, 
  createTheme, 
  CssBaseline,
  Toolbar,
  Typography,
  IconButton,
  FormControl,
  InputLabel,
  Select,
  MenuItem
} from '@mui/material';
import LanguageIcon from '@mui/icons-material/Language';
import PdfProcessingTab from './components/PdfProcessingTab';
import VectorDatabaseTab from './components/VectorDatabaseTab';
import ChatTab from './components/ChatTab';
import DocumentsTab from './components/DocumentsTab';

import FinancialDataTab from './components/FinancialDataTab';
import DataFormulatorTab from './components/DataFormulatorTab';
import axios from 'axios';
import './App.css';

// UI text for multilingual support
const UI_TEXT = {
  en: {
    app_title: "Harmonized Financial Statement RAG Chatbot",
    tab_chat: "Chat",
    tab_docs: "Documents",
    tab_vis: "Vector Visualization",
    tab_fin: "Financial Data",
    tab_form: "Data Formulator",
    language: "Language",
    db_not_found: "Database not found",
    db_create_instructions: "To create the database: 1. Upload PDFs in the 'Documents' tab or 2. Run the scripts 0_pdf_to_text.py and 1_text_to_vector_db.py"
  },
  it: {
    app_title: "Chatbot RAG per Bilanci Armonizzati",
    tab_chat: "Chat",
    tab_docs: "Documenti",
    tab_vis: "Visualizzazione Vettori",
    tab_fin: "Dati Finanziari",
    tab_form: "Data Formulator",
    language: "Lingua",
    db_not_found: "Database non trovato",
    db_create_instructions: "Per creare il database: 1. Carica PDF nella scheda 'Documenti' oppure 2. Esegui gli script 0_pdf_to_text.py e 1_text_to_vector_db.py"
  }
};

// Create a theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
});

function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function App() {
  const [tabValue, setTabValue] = useState(0);
  const [language, setLanguage] = useState('en');
  const [dbStatus, setDbStatus] = useState({ exists: false, loading: true });
  const ui = UI_TEXT[language];

  // Check database status on component mount
  useEffect(() => {
    const checkDatabase = async () => {
      try {
        const response = await axios.get('/check-database');
        setDbStatus({
          exists: response.data.exists,
          loading: false,
          details: response.data
        });
      } catch (error) {
        console.error('Error checking database:', error);
        setDbStatus({
          exists: false,
          loading: false,
          error: error.message
        });
      }
    };
    
    checkDatabase();
  }, []);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleLanguageChange = (event) => {
    setLanguage(event.target.value);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <div className="app">
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              {ui.app_title}
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <IconButton color="inherit">
                <LanguageIcon />
              </IconButton>
              <FormControl variant="outlined" size="small" sx={{ m: 1, minWidth: 120, backgroundColor: 'white', borderRadius: 1 }}>
                <InputLabel id="language-select-label">{ui.language}</InputLabel>
                <Select
                  labelId="language-select-label"
                  id="language-select"
                  value={language}
                  onChange={handleLanguageChange}
                  label={ui.language}
                >
                  <MenuItem value="en">English</MenuItem>
                  <MenuItem value="it">Italiano</MenuItem>
                </Select>
              </FormControl>
            </Box>
          </Toolbar>
          <Tabs 
            value={tabValue} 
            onChange={handleTabChange}
            variant="scrollable"
            scrollButtons="auto"
            textColor="inherit"
            indicatorColor="secondary"
          >
            <Tab label={language === 'it' ? 'Elaborazione PDF' : 'PDF Processing'} />
            <Tab label={language === 'it' ? 'Database Vettoriale' : 'Vector Database'} />
            <Tab label={language === 'it' ? 'Documenti' : 'Documents'} />

            <Tab label={language === 'it' ? 'Chat' : 'Chat'} />
            <Tab label={language === 'it' ? 'Dati Finanziari' : 'Financial Data'} />
            <Tab label={language === 'it' ? 'Data Formulator' : 'Data Formulator'} />
          </Tabs>
        </AppBar>
        
        <Container maxWidth="lg" className="content-container">
          <TabPanel value={tabValue} index={0}>
            <PdfProcessingTab language={language} />
          </TabPanel>
          <TabPanel value={tabValue} index={1}>
            <VectorDatabaseTab language={language} />
          </TabPanel>
          <TabPanel value={tabValue} index={2}>
            <DocumentsTab language={language} ui={ui} dbStatus={dbStatus} />
          </TabPanel>

          <TabPanel value={tabValue} index={3}>
            <ChatTab language={language} ui={ui} dbStatus={dbStatus} />
          </TabPanel>
          <TabPanel value={tabValue} index={4}>
            <FinancialDataTab language={language} ui={ui} />
          </TabPanel>
          <TabPanel value={tabValue} index={5}>
            <DataFormulatorTab language={language} ui={ui} />
          </TabPanel>
        </Container>
      </div>
    </ThemeProvider>
  );
}

export default App;
