import React, { useState, useRef, useEffect } from 'react';
import { 
  TextField, 
  Button, 
  Box, 
  Paper, 
  Typography, 
  Card, 
  CardContent,
  Divider,
  CircularProgress,
  Alert,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  ListSubheader,
  Grid,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  IconButton,
  Tooltip,
  Snackbar
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import KeyIcon from '@mui/icons-material/Key';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import ReactMarkdown from 'react-markdown';
import axios from 'axios';

// UI text for multilingual support for this specific component
const UI_TEXT = {
  en: {
    chat_placeholder: "Ask a question about harmonized financial statements...",
    send_button: "Send",
    source_title: "Source",
    no_messages: "No messages yet. Start a conversation!",
    loading: "Generating answer...",
    rag_settings: "RAG Settings",
    similarity_top_k: "Number of similar paragraphs (Top K)",
    temperature: "Temperature",
    sample_questions: "Sample Questions",
    try_these: "Try these questions:",
    llm_info: "Powered by Groq LLM API",
    model_info: "Using model:",
    model_select: "Select model:",
    provider_select: "Select provider:",
    provider_groq: "Groq (Cloud API)",
    provider_ollama: "Ollama (Local Models)",
    provider_ollama_unavailable: "Ollama (Local Models) - Not Available",
    model_categories: {
      production: "Production Models",
      preview: "Preview Models",
      systems: "Preview Systems",
      ollama: "Local Ollama Models",
      other: "Other Models"
    },
    loading_models: "Loading models...",
    error_loading_models: "Error loading models",
    error_msg: "Error: Could not send message. Please try again.",
    api_key_button: "Set Groq API Key",
    api_key_title: "Set Groq API Key",
    api_key_description: "Enter your Groq API key to enable the chat functionality. You can get a key from https://console.groq.com/",
    api_key_placeholder: "gsk_xxxx...",
    api_key_save: "Save",
    api_key_cancel: "Cancel",
    api_key_success: "API key saved successfully!",
    api_key_error: "Error: Invalid API key",
    api_key_status_valid: "API Key: Valid",
    api_key_status_invalid: "API Key: Not configured"
  },
  it: {
    chat_placeholder: "Fai una domanda sui bilanci armonizzati...",
    send_button: "Invia",
    source_title: "Fonte",
    no_messages: "Nessun messaggio. Inizia una conversazione!",
    loading: "Generazione risposta in corso...",
    rag_settings: "Impostazioni RAG",
    similarity_top_k: "Numero di paragrafi simili (Top K)",
    temperature: "Temperatura",
    sample_questions: "Domande di Esempio",
    try_these: "Prova queste domande:",
    llm_info: "Alimentato da Groq LLM API",
    model_info: "Modello in uso:",
    model_select: "Seleziona modello:",
    provider_select: "Seleziona provider:",
    provider_groq: "Groq (API Cloud)",
    provider_ollama: "Ollama (Modelli Locali)",
    provider_ollama_unavailable: "Ollama (Modelli Locali) - Non Disponibile",
    model_categories: {
      production: "Modelli di Produzione",
      preview: "Modelli di Anteprima",
      systems: "Sistemi di Anteprima",
      ollama: "Modelli Locali Ollama",
      other: "Altri Modelli"
    },
    loading_models: "Caricamento modelli...",
    error_loading_models: "Errore nel caricamento dei modelli",
    error_msg: "Errore: Impossibile inviare il messaggio. Riprova.",
    api_key_button: "Imposta Chiave API Groq",
    api_key_title: "Imposta Chiave API Groq",
    api_key_description: "Inserisci la tua chiave API Groq per abilitare la funzionalità di chat. Puoi ottenere una chiave da https://console.groq.com/",
    api_key_placeholder: "gsk_xxxx...",
    api_key_save: "Salva",
    api_key_cancel: "Annulla",
    api_key_success: "Chiave API salvata con successo!",
    api_key_error: "Errore: Chiave API non valida",
    api_key_status_valid: "Chiave API: Valida",
    api_key_status_invalid: "Chiave API: Non configurata"
  }
};

// Sample questions in both languages
const SAMPLE_QUESTIONS = {
  en: [
    "What is the structure of the harmonized financial statement?",
    "How are cash flows classified in the budget?",
    "What are the main differences between BDAP and traditional accounting?",
    "Can you explain the concept of 'competenza finanziaria potenziata'?",
    "How should I classify capital grants in the harmonized budget?"
  ],
  it: [
    "Qual è la struttura del bilancio armonizzato?",
    "Come vengono classificati i flussi di cassa nel bilancio?",
    "Quali sono le principali differenze tra BDAP e la contabilità tradizionale?",
    "Puoi spiegare il concetto di 'competenza finanziaria potenziata'?",
    "Come devo classificare i contributi in conto capitale nel bilancio armonizzato?"
  ]
};

const ChatTab = ({ language, dbStatus }) => {
  const ui = UI_TEXT[language];
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [similarityTopK, setSimilarityTopK] = useState(5);
  const [temperature, setTemperature] = useState(0.2);
  const [modelInfo, setModelInfo] = useState(null);
  const [selectedModel, setSelectedModel] = useState('');
  const [productionModels, setProductionModels] = useState([]);
  const [previewModels, setPreviewModels] = useState([]);
  const [previewSystems, setPreviewSystems] = useState([]);
  const [defaultModel, setDefaultModel] = useState('');
  const [selectedProvider, setSelectedProvider] = useState('groq');
  const [ollamaModels, setOllamaModels] = useState([]);
  const [ollamaAvailable, setOllamaAvailable] = useState(false);
  const [loadingModels, setLoadingModels] = useState(false);
  const [modelsError, setModelsError] = useState(null);
  const [apiKeyDialogOpen, setApiKeyDialogOpen] = useState(false);
  const [apiKey, setApiKey] = useState('');
  const [apiKeySaving, setApiKeySaving] = useState(false);
  const [apiKeyStatus, setApiKeyStatus] = useState({
    exists: false,
    valid: false,
    model: null
  });
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'info'
  });
  const messagesEndRef = useRef(null);

  // Scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Check API key status and load models on component mount
  useEffect(() => {
    // Fetch models when component mounts
    axios.get('/groq-models')
      .then(response => {
        const allModels = response.data;
        setProductionModels(allModels.production_models || []);
        setPreviewModels(allModels.preview_models || []);
        setPreviewSystems(allModels.preview_systems || []);

        // Check if Ollama is available
        const ollamaIsAvailable = allModels.ollama_available || false;
        setOllamaAvailable(ollamaIsAvailable);

        // Set Ollama models
        const retrievedOllamaModels = allModels.ollama_models || [];
        console.log(`Found ${retrievedOllamaModels.length} Ollama models, available: ${ollamaIsAvailable}`);
        setOllamaModels(retrievedOllamaModels);

        // Set default model
        setDefaultModel(allModels.default_model);

        // If no model is selected yet, set the default
        if (!selectedModel && allModels.default_model) {
          setSelectedModel(allModels.default_model);
          // Default to Groq as provider if we have a default model
          setSelectedProvider('groq');
        }
      })
      .catch(error => {
        console.error('Error fetching models:', error);
        setOllamaAvailable(false);
      });

    // Check API key status
    axios.get('/check-api-key')
      .then(response => {
        setApiKeyStatus({
          exists: response.data.key_exists,
          valid: response.data.client_initialized,
          model: response.data.model
        });

        // If we have a valid model from the API key check, update modelInfo
        if (response.data.model) {
          setModelInfo(response.data.model);
        }
      })
      .catch(error => {
        console.error('Error checking API key status:', error);
      });

    // Set up interval to check API key status periodically
    const intervalId = setInterval(() => {
      axios.get('/check-api-key')
        .then(response => {
          setApiKeyStatus({
            exists: response.data.key_exists,
            valid: response.data.client_initialized,
            model: response.data.model
          });

          // If we have a valid model from the API key check, update modelInfo
          if (response.data.model) {
            setModelInfo(response.data.model);
          }
        })
        .catch(error => {
          console.error('Error checking API key status:', error);
        });
    }, 60000); // Check every minute

    // Clean up interval on unmount
    return () => clearInterval(intervalId);
  }, []);

  // Update model selection when provider changes or when model lists load
  useEffect(() => {
    console.log(`Provider changed to: ${selectedProvider}`);
    console.log(`Available Ollama models: ${ollamaModels.length}, Available: ${ollamaAvailable}`);
    console.log(`Available Groq models: Production=${productionModels.length}, Preview=${previewModels.length}`);
    
    // Reset the selected model when changing provider
    if (selectedProvider === 'groq') {
      // For Groq, start with production models
      if (productionModels.length > 0) {
        console.log(`Setting to first production model: ${productionModels[0]}`);
        setSelectedModel(productionModels[0]);
      } else if (previewModels.length > 0) {
        // Fallback to preview models
        console.log(`Setting to first preview model: ${previewModels[0]}`);
        setSelectedModel(previewModels[0]);
      } else if (previewSystems.length > 0) {
        // Last resort: preview systems
        console.log(`Setting to first preview system: ${previewSystems[0]}`);
        setSelectedModel(previewSystems[0]);
      }
    } else if (selectedProvider === 'ollama') {
      // Only try to select Ollama model if available
      if (ollamaModels.length > 0 && ollamaAvailable) {
        console.log(`Setting to first Ollama model: ${ollamaModels[0]}`);
        setSelectedModel(ollamaModels[0]);
      } else {
        // No Ollama models available, inform user
        console.log('No Ollama models available for selection');
      }
    }
  }, [selectedProvider, productionModels, previewModels, previewSystems, ollamaModels, ollamaAvailable]);

  // Function to check API key status
  const checkApiKeyStatus = async () => {
    try {
      const response = await axios.get('/check-api-key');
      setApiKeyStatus({
        exists: response.data.key_exists,
        valid: response.data.client_initialized,
        model: response.data.model
      });

      // If we have a valid model from the API key check, update modelInfo
      if (response.data.model) {
        setModelInfo(response.data.model);
      }
    } catch (error) {
      console.error('Error checking API key status:', error);
    }
  };

  // Function to handle API key dialog open
  const handleApiKeyDialogOpen = () => {
    setApiKeyDialogOpen(true);
  };

  // Function to handle API key dialog close
  const handleApiKeyDialogClose = () => {
    setApiKeyDialogOpen(false);
    setApiKey('');
  };

  // Function to save API key
  const handleSaveApiKey = async () => {
    // Validate API key format (basic check - accept both sk- and gsk_ formats)
    if (!apiKey) {
      setSnackbar({
        open: true,
        message: ui.api_key_error,
        severity: 'error'
      });
      return;
    }

    setApiKeySaving(true);
    try {
      const response = await axios.post('/set-api-key', {
        api_key: apiKey
      });

      if (response.data.success) {
        setSnackbar({
          open: true,
          message: ui.api_key_success,
          severity: 'success'
        });
        handleApiKeyDialogClose();
        checkApiKeyStatus(); // Refresh API key status
      } else {
        setSnackbar({
          open: true,
          message: response.data.message,
          severity: 'error'
        });
      }
    } catch (error) {
      console.error('Error saving API key:', error);
      setSnackbar({
        open: true,
        message: ui.api_key_error,
        severity: 'error'
      });
    } finally {
      setApiKeySaving(false);
    }
  };

  // Handle snackbar close
  const handleSnackbarClose = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  // Fetch available models from API
  const fetchAvailableModels = async () => {
    setLoadingModels(true);
    try {
      const response = await axios.get('/groq-models');
      setProductionModels(response.data.production_models || []);
      setPreviewModels(response.data.preview_models || []);
      setPreviewSystems(response.data.preview_systems || []);

      // Check if Ollama is available
      const ollamaIsAvailable = response.data.ollama_available || false;
      setOllamaAvailable(ollamaIsAvailable);

      // Set Ollama models
      const retrievedOllamaModels = response.data.ollama_models || [];
      console.log(`Found ${retrievedOllamaModels.length} Ollama models, available: ${ollamaIsAvailable}`);
      setOllamaModels(retrievedOllamaModels);

      // Set default model
      setDefaultModel(response.data.default_model);

      setLoadingModels(false);
    } catch (error) {
      console.error('Error fetching models:', error);
      setLoadingModels(false);
    }
  };

  // Handle model change
  const handleModelChange = (event) => {
    const selectedModelValue = event.target.value;
    console.log(`Setting selected model to: ${selectedModelValue}`);
    setSelectedModel(selectedModelValue);
    
    // Add additional debug info
    if (selectedProvider === 'ollama') {
      console.log(`Selected an Ollama model with provider: ${selectedProvider}`);
    } else {
      console.log(`Selected a Groq model with provider: ${selectedProvider}`);
    }
  };

  // Function to handle provider selection
  const selectProvider = (provider) => {
    console.log(`Selecting provider: ${provider}`);
    
    // Set the provider first
    setSelectedProvider(provider);
    
    // Use setTimeout to ensure the provider change has been processed
    setTimeout(() => {
      // Reset the selected model when changing provider
      if (provider === 'groq') {
        // For Groq, start with production models
        if (productionModels.length > 0) {
          console.log(`Setting to Groq production model: ${productionModels[0]}`);
          setSelectedModel(productionModels[0]);
        } else if (previewModels.length > 0) {
          // Fallback to preview models
          console.log(`Setting to Groq preview model: ${previewModels[0]}`);
          setSelectedModel(previewModels[0]);
        } else if (previewSystems.length > 0) {
          // Last resort: preview systems
          console.log(`Setting to Groq preview system: ${previewSystems[0]}`);
          setSelectedModel(previewSystems[0]);
        }
      } else if (provider === 'ollama') {
        // Only try to select Ollama model if available
        if (ollamaModels.length > 0 && ollamaAvailable) {
          console.log(`Setting to Ollama model: ${ollamaModels[0]}`);
          setSelectedModel(ollamaModels[0]);
        } else {
          // No Ollama models available, warn user
          console.log('No Ollama models available for selection');
          // Still set an empty model to clear any previous selection
          setSelectedModel('');
        }
      }
    }, 50); // Small delay to ensure state updates properly
  };

  // Handle provider change (Groq or Ollama)
  const handleProviderChange = (event) => {
    // If called from Select component, get value from event
    if (event && event.target && event.target.value) {
      selectProvider(event.target.value);
    }
    // If called directly with a string value
    else if (typeof event === 'string') {
      selectProvider(event);
    }
  };

  const handleQueryChange = (e) => {
    setQuery(e.target.value);
  };

  const handleSimilarityTopKChange = (event, newValue) => {
    setSimilarityTopK(newValue);
  };

  const handleTemperatureChange = (event, newValue) => {
    setTemperature(newValue);
  };

  const handleSampleQuestionClick = (question) => {
    setQuery(question);
    // Optional: automatically submit the question
    setTimeout(() => {
      handleSubmit({ preventDefault: () => {} }, question);
    }, 100);
  };

  const handleSubmit = async (e, forcedQuery = null) => {
    e && e.preventDefault();
    const queryText = forcedQuery || query;
    if (!queryText.trim()) return;
    
    // Check if we have a model selected
    if (!selectedModel) {
      setError("Please select a model first");
      return;
    }
    
    // Clear previous error
    setError(null);
    
    // Log submission details
    console.log(`Submitting query with model: ${selectedModel}`);
    console.log(`Provider: ${selectedProvider}, TopK: ${similarityTopK}, Temp: ${temperature}`);
    
    // Add user message
    const userMessage = {
      sender: 'user',
      text: queryText,
      timestamp: new Date().toISOString()
    };
    setMessages(prevMessages => [...prevMessages, userMessage]);
    
    // Add loading message
    const loadingMessage = {
      sender: 'bot',
      text: ui.loading,
      timestamp: new Date().toISOString(),
      isLoading: true
    };
    setMessages(prevMessages => [...prevMessages, loadingMessage]);
    
    // Reset query input
    setQuery('');
    
    // Scroll to the loading message
    scrollToBottom();
    
    try {
      // Prepare request payload
      const payload = {
        query: queryText,
        similarity_top_k: similarityTopK,
        temperature: temperature,
        model: selectedModel,
        provider: selectedProvider // Add provider info
      };
      
      console.log('Sending request with payload:', JSON.stringify(payload));
      
      // Use dedicated Ollama endpoint if using Ollama provider
      let endpoint = '/chat';
      if (selectedProvider === 'ollama') {
        endpoint = '/ollama-chat';
        console.log('Using dedicated Ollama endpoint');
      }
      
      // Send query to the appropriate API endpoint
      const response = await axios.post(endpoint, payload);
      
      console.log('Received response:', response.data);
      
      // Remove loading message
      setMessages(prevMessages => prevMessages.filter(msg => !msg.isLoading));
      
      // Add bot response
      const botMessage = {
        sender: 'bot',
        text: response.data.answer,
        timestamp: new Date().toISOString(),
        sources: response.data.sources || [],
        model: response.data.model || null
      };

      // Update model info if available
      if (response.data.model) {
        setModelInfo(response.data.model);
        console.log(`Updated model info to: ${response.data.model}`);
      }
      setMessages(prevMessages => [...prevMessages, botMessage]);

      // Scroll to the answer
      scrollToBottom();
    } catch (error) {
      // Remove loading message
      setMessages(prevMessages => prevMessages.filter(msg => !msg.isLoading));

      // Show a more detailed error message if available
      const errorMessage = error.response?.data?.detail || ui.error_msg;
      setError(errorMessage);
      
      // Log error details
      console.error('Error sending message:', error);
      if (error.response) {
        console.error('Response data:', error.response.data);
        console.error('Response status:', error.response.status);
      }
    }
  };

  useEffect(() => {
    // Check API key status
    axios.get('/check-api-key')
      .then(response => {
        setApiKeyStatus({
          exists: response.data.key_exists,
          valid: response.data.client_initialized,
          model: response.data.model
        });

        // If we have a valid model from the API key check, update modelInfo
        if (response.data.model) {
          setModelInfo(response.data.model);
        }
      })
      .catch(error => {
        console.error('Error checking API key status:', error);
      });

    // Fetch models
    fetchAvailableModels();

    // Log initial state for debugging
    console.log('Initial render - setting up component');
  }, []);

  return (
    <Box className="chat-container">
      <Grid container spacing={2}>
        <Grid item xs={12} md={9}>
          {/* Sample Questions Area */}
          <Paper elevation={1} sx={{ p: 2, mb: 2, bgcolor: '#f9f9f9' }}>
            <Typography variant="h6" gutterBottom>
              {ui.sample_questions}
            </Typography>
            <Typography variant="body2" gutterBottom>
              {ui.try_these}
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
              {SAMPLE_QUESTIONS[language].map((question, index) => (
                <Button 
                  key={index} 
                  variant="outlined" 
                  size="small"
                  onClick={() => handleSampleQuestionClick(question)}
                  sx={{ 
                    textTransform: 'none', 
                    borderColor: '#bbdefb',
                    color: '#1976d2',
                    '&:hover': {
                      backgroundColor: '#e3f2fd',
                      borderColor: '#2196f3'
                    }
                  }}
                >
                  {question.length > 60 ? question.substring(0, 57) + '...' : question}
                </Button>
              ))}
            </Box>
          </Paper>

          {/* Chat Messages Area */}
          <Paper
            elevation={3}
            sx={{
              height: 'calc(100vh - 350px)',
              overflow: 'auto',
              p: 2,
              mb: 2,
              bgcolor: '#f8f9fa'
            }}
            ref={messagesEndRef}
          >
            {/* No messages state */}
            {messages.length === 0 && (
              <Box
                sx={{
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  height: '100%',
                  color: 'text.secondary',
                  flexDirection: 'column'
                }}
              >
                <Typography variant="body1" sx={{ mb: 2 }}>
                  {ui.no_messages}
                </Typography>
              </Box>
            )}

            {/* Display messages */}
            {messages.map((msg, index) => (
              <Box key={index} sx={{ mb: 3 }}>
                {/* Message */}
                <Card 
                  sx={{
                    bgcolor: msg.sender === 'user' ? '#e3f2fd' : '#fff',
                    maxWidth: '90%',
                    ml: msg.sender === 'user' ? 'auto' : 0,
                    mr: msg.sender === 'user' ? 0 : 'auto',
                    mb: 1,
                    boxShadow: msg.sender === 'user' ? 1 : 2 
                  }}
                >
                  <CardContent>
                    <Typography variant="body1" component="div">
                      {msg.sender === 'bot' ? (
                        <ReactMarkdown>{msg.text}</ReactMarkdown>
                      ) : (
                        msg.text
                      )}
                    </Typography>

                    {/* Display sources for bot messages */}
                    {msg.sender === 'bot' && msg.sources && msg.sources.length > 0 && (
                      <Box sx={{ mt: 2 }}>
                        <Divider sx={{ mb: 1 }} />
                        <Typography variant="subtitle2" color="primary" fontWeight="medium">
                          {ui.source_title}:
                        </Typography>
                        {/* Group sources by document */}
                        {Object.entries(msg.sources.reduce((acc, source) => {
                          const sourceKey = source.source;
                          if (!acc[sourceKey]) acc[sourceKey] = [];
                          acc[sourceKey].push(source);
                          return acc;
                        }, {})).map(([sourceName, sourceGroup], groupIndex) => (
                          <Box key={groupIndex} sx={{ mt: 1, mb: 2 }}>
                            <Typography variant="body2" sx={{ 
                              fontWeight: 'bold', 
                              fontSize: '0.85rem',
                              color: '#1976d2'
                            }}>
                              {sourceName}
                            </Typography>
                            {sourceGroup.map((source, idx) => (
                              <Box key={idx} sx={{ 
                                mt: 0.5,
                                bgcolor: '#f5f5f5', 
                                p: 1, 
                                borderRadius: 1,
                                borderLeft: '3px solid #1976d2'
                              }}>
                                <Typography variant="body2" sx={{ 
                                  fontSize: '0.8rem', 
                                  fontStyle: 'italic',
                                  display: 'flex',
                                  justifyContent: 'space-between'
                                }}>
                                  <span>{source.page ? `Page ${source.page}` : source.file}</span>
                                  <span style={{color: '#666'}}>{source.similarity ? `${Math.round(source.similarity * 100)}% match` : ''}</span>
                                </Typography>
                                <Typography variant="body2" sx={{ 
                                  fontSize: '0.8rem',
                                  mt: 0.5,
                                  maxHeight: '80px',
                                  overflow: 'auto'
                                }}>
                                  {source.text.length > 150 ? 
                                    `${source.text.substring(0, 150)}...` : 
                                    source.text}
                                </Typography>
                              </Box>
                            ))}
                          </Box>
                        ))}
                      </Box>
                    )}
                  </CardContent>
                </Card>

                {/* Timestamp and model info */}
                <Box sx={{ 
                  display: 'flex', 
                  justifyContent: msg.sender === 'user' ? 'flex-end' : 'flex-start',
                  alignItems: 'center',
                  mr: msg.sender === 'user' ? 1 : 0,
                  ml: msg.sender === 'user' ? 0 : 1,
                }}>
                  <Typography variant="caption">
                    {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </Typography>

                  {/* Show model used for responses */}
                  {msg.sender === 'bot' && msg.model && (
                    <Typography variant="caption" sx={{ ml: 1, color: 'text.secondary', fontSize: '0.7rem' }}>
                      · {msg.model}
                    </Typography>
                  )}
                </Box>
              </Box>
            ))}
            <div ref={messagesEndRef} />
            {loading && (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
                <CircularProgress size={24} sx={{ mr: 1 }} />
                <Typography variant="body2">{ui.loading}</Typography>
              </Box>
            )}
            {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
          </Paper>

          <Box component="form" onSubmit={handleSubmit} sx={{ mt: 2, display: 'flex' }}>
            <TextField
              fullWidth
              variant="outlined"
              placeholder={ui.chat_placeholder}
              value={query}
              onChange={handleQueryChange}
              disabled={loading || !dbStatus.exists}
              sx={{ mr: 1 }}
            />
            <Button 
              variant="contained" 
              color="primary" 
              type="submit" 
              disabled={loading || !query.trim() || !dbStatus.exists}
              endIcon={<SendIcon />}
            >
              {ui.send_button}
            </Button>
          </Box>
        </Grid>

        <Grid item xs={12} md={3}>
          <Paper elevation={0} sx={{ p: 2, mb: 2, bgcolor: '#f9f9f9', borderRadius: 2 }}>
            {/* Provider Selection - First Level */}
            {loadingModels ? (
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <CircularProgress size={20} sx={{ mr: 1 }} />
                <Typography variant="body2">{ui.loading_models}</Typography>
              </Box>
            ) : modelsError ? (
              <Alert severity="error" sx={{ mb: 2 }}>{ui.error_loading_models}</Alert>
            ) : (
              <>
                {/* Provider Selection (Groq or Ollama) */}
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel id="provider-select-label">Select Provider:</InputLabel>
                  <Select
                    labelId="provider-select-label"
                    id="provider-select"
                    value={selectedProvider}
                    label="Select Provider"
                    onChange={handleProviderChange}
                    disabled={!dbStatus.exists}
                  >
                    <MenuItem value="groq" onClick={() => selectProvider('groq')}>Groq (Cloud API)</MenuItem>
                    <MenuItem 
                      value="ollama" 
                      onClick={() => selectProvider('ollama')}
                      disabled={!ollamaAvailable || ollamaModels.length === 0}
                    >
                      {ollamaAvailable ? 'Ollama (Local Models)' : 'Ollama (Local Models) - Not Available'}
                    </MenuItem>
                  </Select>
                </FormControl>

                {/* Model Selection - Second Level */}
                {/* Simplified Model Selection - Second Level */}
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel id="model-select-label">{ui.model_select}</InputLabel>
                  
                  {/* GROQ MODELS */}
                  {selectedProvider === 'groq' && (
                    <Select
                      labelId="model-select-label"
                      id="model-select"
                      value={selectedModel}
                      label={ui.model_select}
                      onChange={handleModelChange}
                      disabled={!dbStatus.exists}
                    >
                      {/* Production Models */}
                      {productionModels.length > 0 && [
                        <ListSubheader key="production-header">{ui.model_categories.production}</ListSubheader>,
                        ...productionModels.map(model => (
                          <MenuItem key={model} value={model}>{model}</MenuItem>
                        ))
                      ]}

                      {/* Preview Models */}
                      {previewModels.length > 0 && [
                        <ListSubheader key="preview-header">{ui.model_categories.preview}</ListSubheader>,
                        ...previewModels.map(model => (
                          <MenuItem key={model} value={model}>{model}</MenuItem>
                        ))
                      ]}

                      {/* Preview Systems */}
                      {previewSystems.length > 0 && [
                        <ListSubheader key="systems-header">{ui.model_categories.systems}</ListSubheader>,
                        ...previewSystems.map(model => (
                          <MenuItem key={model} value={model}>{model}</MenuItem>
                        ))
                      ]}
                    </Select>
                  )}
                  
                  {/* OLLAMA MODELS */}
                  {selectedProvider === 'ollama' && (
                    <Select
                      labelId="model-select-label"
                      id="model-select-ollama"
                      value={selectedModel}
                      label={ui.model_select}
                      onChange={handleModelChange}
                      disabled={!dbStatus.exists || !ollamaAvailable || ollamaModels.length === 0}
                    >
                      <ListSubheader>{ui.model_categories.ollama}</ListSubheader>
                      {ollamaModels.map(model => (
                        <MenuItem key={model} value={model}>{model}</MenuItem>
                      ))}
                      {ollamaModels.length === 0 && (
                        <MenuItem disabled>No Ollama models available</MenuItem>
                      )}
                    </Select>
                  )}
                </FormControl>
              </>
            )}
          </Paper>
          
          {/* LLM Model Info */}
          <Paper elevation={1} sx={{ p: 2, bgcolor: '#f5f5f5', mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <Typography variant="subtitle2" sx={{ color: '#666' }}>
                {ui.llm_info}
              </Typography>
              <Tooltip title={ui.api_key_button}>
                <IconButton 
                  size="small" 
                  onClick={handleApiKeyDialogOpen}
                  color={apiKeyStatus.valid ? "primary" : "default"}
                >
                  <KeyIcon />
                </IconButton>
              </Tooltip>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
              {apiKeyStatus.valid ? (
                <CheckCircleIcon fontSize="small" color="success" sx={{ mr: 1 }} />
              ) : (
                <ErrorIcon fontSize="small" color="error" sx={{ mr: 1 }} />
              )}
              <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
                {apiKeyStatus.valid ? ui.api_key_status_valid : ui.api_key_status_invalid}
              </Typography>
            </Box>
            {modelInfo && (
              <Typography variant="body2" sx={{ mt: 1, fontFamily: 'monospace', fontSize: '0.8rem' }}>
                {ui.model_info} {modelInfo}
              </Typography>
            )}
          </Paper>
        </Grid>
      </Grid>
      
      {/* API Key Dialog */}
      <Dialog open={apiKeyDialogOpen} onClose={handleApiKeyDialogClose} fullWidth maxWidth="sm">
        <DialogTitle>
          {ui.api_key_title}
        </DialogTitle>
        <DialogContent>
          <DialogContentText sx={{ mb: 2 }}>
            {ui.api_key_description}
          </DialogContentText>
          <TextField
            autoFocus
            margin="dense"
            label="Groq API Key"
            type="password"
            fullWidth
            variant="outlined"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder={ui.api_key_placeholder}
            disabled={apiKeySaving}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={handleApiKeyDialogClose} disabled={apiKeySaving}>
            {ui.api_key_cancel}
          </Button>
          <Button 
            onClick={handleSaveApiKey} 
            variant="contained" 
            color="primary"
            disabled={apiKeySaving || !apiKey.trim()}
          >
            {apiKeySaving ? <CircularProgress size={24} /> : ui.api_key_save}
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleSnackbarClose}
        message={snackbar.message}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      />
    </Box>
  );
};

export default ChatTab;
