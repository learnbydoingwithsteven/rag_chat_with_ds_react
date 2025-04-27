import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Alert,
  AlertTitle,
  CircularProgress,
  Divider,
  Link,
  Card,
  CardContent,
  CardActions,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  TextField,
  IconButton
} from '@mui/material';
import LaunchIcon from '@mui/icons-material/Launch';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import axios from 'axios';

// UI text for multilingual support for this specific component
const UI_TEXT = {
  en: {
    title: "Data Formulator Integration",
    description: "Data Formulator is a powerful tool for analyzing and visualizing financial data. You can use it alongside this application to explore the data you've downloaded.",
    checking: "Checking if Data Formulator is installed...",
    not_installed: "Data Formulator is not installed",
    installed: "Data Formulator is installed",
    install_instructions: "To install Data Formulator, run the following command in your terminal:",
    install_command: "pip install data-formulator",
    launch_title: "Launch Data Formulator",
    launch_description: "Click the button below to launch Data Formulator in a new window.",
    launching: "Launching Data Formulator...",
    launch_button: "Launch Data Formulator",
    launch_error: "Failed to launch Data Formulator. Please try the manual command below.",
    manual_launch: "Alternatively, you can run Data Formulator manually by executing this command in your terminal:",
    manual_command: "python -m data_formulator",
    copy_command: "Copy command",
    command_copied: "Command copied to clipboard!",
    usage_title: "Using Data Formulator with the Financial Data",
    usage_steps: [
      "Download CSV files from the Financial Data tab",
      "Launch Data Formulator using the button above",
      "In Data Formulator, click 'Load Data' and select the CSV files you downloaded",
      "Use Data Formulator's tools to analyze and visualize the data"
    ],
    resources_title: "Additional Resources",
    documentation: "Data Formulator Documentation",
    refresh_button: "Check Again",
    launch_success: "Data Formulator launched successfully! Look for a new window that has opened."
  },
  it: {
    title: "Integrazione Data Formulator",
    description: "Data Formulator è uno strumento potente per analizzare e visualizzare dati finanziari. Puoi utilizzarlo insieme a questa applicazione per esplorare i dati che hai scaricato.",
    checking: "Controllo se Data Formulator è installato...",
    not_installed: "Data Formulator non è installato",
    installed: "Data Formulator è installato",
    install_instructions: "Per installare Data Formulator, esegui il seguente comando nel tuo terminale:",
    install_command: "pip install data-formulator",
    launch_title: "Avvia Data Formulator",
    launch_description: "Clicca il pulsante sotto per avviare Data Formulator in una nuova finestra.",
    launching: "Avvio Data Formulator...",
    launch_button: "Avvia Data Formulator",
    launch_error: "Impossibile avviare Data Formulator. Prova il comando manuale sotto.",
    manual_launch: "In alternativa, puoi avviare Data Formulator manualmente eseguendo questo comando nel tuo terminale:",
    manual_command: "python -m data_formulator",
    copy_command: "Copia comando",
    command_copied: "Comando copiato negli appunti!",
    usage_title: "Utilizzo di Data Formulator con i Dati Finanziari",
    usage_steps: [
      "Scarica file CSV dalla scheda Dati Finanziari",
      "Avvia Data Formulator usando il pulsante sopra",
      "In Data Formulator, clicca 'Carica Dati' e seleziona i file CSV che hai scaricato",
      "Utilizza gli strumenti di Data Formulator per analizzare e visualizzare i dati"
    ],
    resources_title: "Risorse Aggiuntive",
    documentation: "Documentazione Data Formulator",
    refresh_button: "Controlla di Nuovo",
    launch_success: "Data Formulator avviato con successo! Cerca una nuova finestra che si è aperta."
  }
};

const DataFormulatorTab = ({ language }) => {
  const ui = UI_TEXT[language];
  const [checking, setChecking] = useState(true);
  const [installed, setInstalled] = useState(false);
  const [launching, setLaunching] = useState(false);
  const [launchResult, setLaunchResult] = useState(null);
  const [commandCopied, setCommandCopied] = useState(false);

  useEffect(() => {
    checkDataFormulatorInstallation();
  }, []);

  const checkDataFormulatorInstallation = async () => {
    setChecking(true);
    try {
      const response = await axios.get('/check-data-formulator');
      setInstalled(response.data.installed);
    } catch (error) {
      console.error('Error checking Data Formulator installation:', error);
      setInstalled(false);
    } finally {
      setChecking(false);
    }
  };

  const handleRefresh = () => {
    checkDataFormulatorInstallation();
    setLaunchResult(null);
  };

  const launchDataFormulator = async () => {
    setLaunching(true);
    setLaunchResult(null);
    
    try {
      const response = await axios.post('/launch-data-formulator');
      setLaunchResult({
        success: response.data.success,
        message: response.data.message
      });
    } catch (error) {
      console.error('Error launching Data Formulator:', error);
      setLaunchResult({
        success: false,
        message: ui.launch_error
      });
    } finally {
      setLaunching(false);
    }
  };

  const copyCommand = (command) => {
    navigator.clipboard.writeText(command)
      .then(() => {
        setCommandCopied(true);
        setTimeout(() => setCommandCopied(false), 2000);
      })
      .catch(err => console.error('Failed to copy command:', err));
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        {ui.title}
      </Typography>
      <Typography variant="body1" paragraph>
        {ui.description}
      </Typography>

      {/* Installation Status */}
      <Paper elevation={1} sx={{ p: 2, mb: 3 }}>
        {checking ? (
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <CircularProgress size={24} sx={{ mr: 1.5 }} />
            <Typography>{ui.checking}</Typography>
          </Box>
        ) : installed ? (
          <Alert severity="success" icon={<CheckCircleOutlineIcon />}>
            <AlertTitle>{ui.installed}</AlertTitle>
            {ui.launch_description}
            <Box sx={{ mt: 2 }}>
              <Button
                variant="contained"
                color="primary"
                onClick={launchDataFormulator}
                disabled={launching}
                startIcon={launching ? <CircularProgress size={20} /> : <LaunchIcon />}
              >
                {launching ? ui.launching : ui.launch_button}
              </Button>
              
              {launchResult && (
                <Alert 
                  severity={launchResult.success ? "success" : "error"}
                  sx={{ mt: 2 }}
                >
                  {launchResult.message}
                </Alert>
              )}
              
              <Divider sx={{ my: 2 }} />
              
              <Typography variant="subtitle2" gutterBottom>
                {ui.manual_launch}
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', bgcolor: '#f5f5f5', p: 1, borderRadius: 1 }}>
                <Typography variant="code" sx={{ flex: 1, fontFamily: 'monospace' }}>
                  {ui.manual_command}
                </Typography>
                <IconButton 
                  size="small" 
                  onClick={() => copyCommand(ui.manual_command)}
                  color={commandCopied ? "success" : "default"}
                >
                  {commandCopied ? <CheckCircleOutlineIcon /> : <ContentCopyIcon />}
                </IconButton>
              </Box>
              {commandCopied && (
                <Typography variant="caption" sx={{ display: 'block', mt: 0.5, color: 'success.main' }}>
                  {ui.command_copied}
                </Typography>
              )}
            </Box>
          </Alert>
        ) : (
          <Alert severity="warning" icon={<ErrorOutlineIcon />}>
            <AlertTitle>{ui.not_installed}</AlertTitle>
            <Typography paragraph>
              {ui.install_instructions}
            </Typography>
            <Box sx={{ bgcolor: '#f5f5f5', p: 1, borderRadius: 1, display: 'flex', alignItems: 'center' }}>
              <Typography variant="code" sx={{ flex: 1, fontFamily: 'monospace' }}>
                {ui.install_command}
              </Typography>
              <IconButton 
                size="small" 
                onClick={() => copyCommand(ui.install_command)}
                color={commandCopied ? "success" : "default"}
              >
                {commandCopied ? <CheckCircleOutlineIcon /> : <ContentCopyIcon />}
              </IconButton>
            </Box>
            {commandCopied && (
              <Typography variant="caption" sx={{ display: 'block', mt: 0.5, color: 'success.main' }}>
                {ui.command_copied}
              </Typography>
            )}
            <Button 
              variant="outlined" 
              color="primary" 
              sx={{ mt: 2 }} 
              onClick={handleRefresh}
            >
              {ui.refresh_button}
            </Button>
          </Alert>
        )}
      </Paper>

      {/* Usage Guide */}
      <Paper elevation={1} sx={{ p: 2, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          {ui.usage_title}
        </Typography>
        <Stepper orientation="vertical" sx={{ mt: 2 }}>
          {ui.usage_steps.map((step, index) => (
            <Step key={index} active={true}>
              <StepLabel>{step}</StepLabel>
            </Step>
          ))}
        </Stepper>
      </Paper>

      {/* Additional Resources */}
      <Paper elevation={1} sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          {ui.resources_title}
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column' }}>
          <Link 
            href="https://github.com/example/data-formulator/docs" 
            target="_blank" 
            rel="noopener noreferrer"
            sx={{ mb: 1 }}
          >
            {ui.documentation}
          </Link>
        </Box>
      </Paper>
    </Box>
  );
};

export default DataFormulatorTab;
