import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  Alert,
  Radio,
  RadioGroup,
  FormControlLabel,
  FormControl,
  FormLabel,
  Card,
  CardContent,
  CardActions,
  Pagination,
  Chip,
  Link,
  Tabs,
  Tab,
  IconButton
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import DownloadIcon from '@mui/icons-material/Download';
import InfoIcon from '@mui/icons-material/Info';
import axios from 'axios';

// UI text for multilingual support for this specific component
const UI_TEXT = {
  en: {
    title: "Financial Data Access",
    description: "Access Italian government financial datasets from BDAP (Banca Dati Amministrazioni Pubbliche).",
    api_selection: "API Selection",
    primary_api: "Primary CKAN API",
    alternative_api: "Alternative REST API",
    search_datasets: "Search Datasets",
    search_placeholder: "Enter keywords to search for datasets...",
    search_button: "Search",
    loading_datasets: "Loading datasets...",
    no_datasets: "No datasets found. Try a different search term.",
    dataset_name: "Dataset Name",
    dataset_description: "Description",
    dataset_resources: "Resources",
    dataset_actions: "Actions",
    view_details: "View Details",
    download_csv: "Download CSV",
    loading_resources: "Loading resources...",
    no_resources: "No resources found for this dataset.",
    resource_name: "Resource Name",
    resource_format: "Format",
    resource_size: "Size",
    view_resource: "View",
    download_resource: "Download",
    dataset_details: "Dataset Details",
    visualization: "Visualization",
    raw_data: "Raw Data",
    load_datasets: "Load Available Datasets",
    loading_alt_datasets: "Loading available datasets from the alternative API...",
    no_alt_datasets: "No datasets found in the alternative API.",
    dataset_filter: "Filter datasets",
    dataset_id: "Dataset ID",
    dataset_title: "Title",
    dataset_type: "Type",
    external_resources: "External resources",
    external_resources_info: "You can also access the data directly from the BDAP website:",
    bdap_opendata: "BDAP OpenData Portal",
    loading_error: "Error loading data. Please try again."
  },
  it: {
    title: "Accesso ai Dati Finanziari",
    description: "Accedi ai dataset finanziari del governo italiano da BDAP (Banca Dati Amministrazioni Pubbliche).",
    api_selection: "Selezione API",
    primary_api: "API CKAN Primaria",
    alternative_api: "API REST Alternativa",
    search_datasets: "Cerca Dataset",
    search_placeholder: "Inserisci parole chiave per cercare dataset...",
    search_button: "Cerca",
    loading_datasets: "Caricamento dataset...",
    no_datasets: "Nessun dataset trovato. Prova un termine di ricerca diverso.",
    dataset_name: "Nome Dataset",
    dataset_description: "Descrizione",
    dataset_resources: "Risorse",
    dataset_actions: "Azioni",
    view_details: "Vedi Dettagli",
    download_csv: "Scarica CSV",
    loading_resources: "Caricamento risorse...",
    no_resources: "Nessuna risorsa trovata per questo dataset.",
    resource_name: "Nome Risorsa",
    resource_format: "Formato",
    resource_size: "Dimensione",
    view_resource: "Visualizza",
    download_resource: "Scarica",
    dataset_details: "Dettagli Dataset",
    visualization: "Visualizzazione",
    raw_data: "Dati Grezzi",
    load_datasets: "Carica Dataset Disponibili",
    loading_alt_datasets: "Caricamento dataset disponibili dall'API alternativa...",
    no_alt_datasets: "Nessun dataset trovato nell'API alternativa.",
    dataset_filter: "Filtra dataset",
    dataset_id: "ID Dataset",
    dataset_title: "Titolo",
    dataset_type: "Tipo",
    external_resources: "Risorse esterne",
    external_resources_info: "Puoi anche accedere ai dati direttamente dal sito web BDAP:",
    bdap_opendata: "Portale OpenData BDAP",
    loading_error: "Errore durante il caricamento dei dati. Riprova."
  }
};

const FinancialDataTab = ({ language }) => {
  const ui = UI_TEXT[language];
  const [apiType, setApiType] = useState('primary');
  const [searchQuery, setSearchQuery] = useState('');
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [resources, setResources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [page, setPage] = useState(1);
  const [tabValue, setTabValue] = useState(0);
  const [csvData, setCsvData] = useState(null);
  const [resourceLoading, setResourceLoading] = useState(false);
  
  // Alternative API state
  const [loadedAltDatasets, setLoadedAltDatasets] = useState(false);
  const [filteredAltDatasets, setFilteredAltDatasets] = useState([]);
  const [altDatasetFilter, setAltDatasetFilter] = useState('');
  const [selectedAltDataset, setSelectedAltDataset] = useState(null);
  const [altDatasetDetails, setAltDatasetDetails] = useState(null);
  
  const rowsPerPage = 10;

  const handleApiTypeChange = (event) => {
    setApiType(event.target.value);
    resetState();
  };

  const resetState = () => {
    setDatasets([]);
    setSelectedDataset(null);
    setResources([]);
    setCsvData(null);
    setPage(1);
    setTabValue(0);
    setSearchQuery('');
    setSelectedAltDataset(null);
    setAltDatasetDetails(null);
    setFilteredAltDatasets([]);
  };

  const handleSearchQueryChange = (event) => {
    setSearchQuery(event.target.value);
  };

  const handleAltDatasetFilterChange = (event) => {
    setAltDatasetFilter(event.target.value);
    if (loadedAltDatasets && datasets.length > 0) {
      filterAltDatasets(event.target.value);
    }
  };

  const filterAltDatasets = (query) => {
    if (!query) {
      setFilteredAltDatasets(datasets);
      return;
    }
    
    const filtered = datasets.filter(dataset => {
      const searchTerms = query.toLowerCase().split(' ');
      const datasetText = (dataset.id + ' ' + dataset.title).toLowerCase();
      return searchTerms.every(term => datasetText.includes(term));
    });
    
    setFilteredAltDatasets(filtered);
  };

  const handleSearch = async () => {
    try {
      setLoading(true);
      setError(null);
      setSelectedDataset(null);
      setCsvData(null);
      
      const response = await axios.get(`/bdap/datasets?q=${searchQuery}`);
      setDatasets(response.data.datasets || []);
      setPage(1);
    } catch (error) {
      console.error('Error searching datasets:', error);
      setError(ui.loading_error);
      setDatasets([]);
    } finally {
      setLoading(false);
    }
  };

  const loadAltDatasets = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await axios.get('/bdap/alternative/datasets');
      setDatasets(response.data.datasets || []);
      setFilteredAltDatasets(response.data.datasets || []);
      setLoadedAltDatasets(true);
    } catch (error) {
      console.error('Error loading alternative datasets:', error);
      setError(ui.loading_error);
      setDatasets([]);
      setFilteredAltDatasets([]);
    } finally {
      setLoading(false);
    }
  };

  const handleDatasetSelect = async (dataset) => {
    setSelectedDataset(dataset);
    setTabValue(0);
    setCsvData(null);
    
    if (apiType === 'primary') {
      try {
        setResourceLoading(true);
        // Get resources for the selected dataset
        const resources = dataset.resources || [];
        setResources(resources);
      } catch (error) {
        console.error('Error fetching resources:', error);
        setResources([]);
      } finally {
        setResourceLoading(false);
      }
    }
  };

  const handleAltDatasetSelect = async (dataset) => {
    setSelectedAltDataset(dataset);
    setAltDatasetDetails(null);
    setCsvData(null);
    
    try {
      setResourceLoading(true);
      // Get details for the selected alternative dataset
      const response = await axios.get(`/bdap/alternative/dataset/${dataset.id}`);
      setAltDatasetDetails(response.data);
    } catch (error) {
      console.error('Error fetching alternative dataset details:', error);
      setAltDatasetDetails(null);
    } finally {
      setResourceLoading(false);
    }
  };

  const handleResourceSelect = async (resource) => {
    try {
      setResourceLoading(true);
      setCsvData(null);
      
      // Fetch CSV data for the selected resource
      const response = await axios.get(`/bdap/resource/${resource.id}`);
      // This would typically return CSV data which we would parse and display
      // For this example, we'll use a placeholder
      setCsvData({
        columns: ['Column A', 'Column B', 'Column C'],
        rows: [
          ['Value 1A', 'Value 1B', 'Value 1C'],
          ['Value 2A', 'Value 2B', 'Value 2C'],
          ['Value 3A', 'Value 3B', 'Value 3C']
        ]
      });
      
      setTabValue(1); // Switch to visualization tab
    } catch (error) {
      console.error('Error fetching resource data:', error);
      setCsvData(null);
    } finally {
      setResourceLoading(false);
    }
  };

  const handlePageChange = (event, value) => {
    setPage(value);
  };

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  // Function to download resource
  const downloadResource = (resource) => {
    window.open(resource.url, '_blank');
  };

  // Render the dataset list for the primary API
  const renderPrimaryDatasetList = () => {
    if (loading) {
      return (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
          <CircularProgress />
          <Typography variant="body2" sx={{ ml: 2 }}>
            {ui.loading_datasets}
          </Typography>
        </Box>
      );
    }

    if (error) {
      return (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      );
    }

    if (datasets.length === 0) {
      return (
        <Typography variant="body2" sx={{ p: 2, textAlign: 'center' }}>
          {ui.no_datasets}
        </Typography>
      );
    }

    // Calculate pagination
    const startIndex = (page - 1) * rowsPerPage;
    const endIndex = startIndex + rowsPerPage;
    const paginatedDatasets = datasets.slice(startIndex, endIndex);
    const totalPages = Math.ceil(datasets.length / rowsPerPage);

    return (
      <>
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>{ui.dataset_name}</TableCell>
                <TableCell>{ui.dataset_description}</TableCell>
                <TableCell align="center">{ui.dataset_resources}</TableCell>
                <TableCell align="center">{ui.dataset_actions}</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {paginatedDatasets.map((dataset) => (
                <TableRow 
                  key={dataset.id}
                  hover
                  selected={selectedDataset?.id === dataset.id}
                  onClick={() => handleDatasetSelect(dataset)}
                  sx={{ cursor: 'pointer' }}
                >
                  <TableCell>{dataset.title}</TableCell>
                  <TableCell>
                    {dataset.notes
                      ? dataset.notes.length > 100
                        ? `${dataset.notes.substring(0, 100)}...`
                        : dataset.notes
                      : ''}
                  </TableCell>
                  <TableCell align="center">
                    {dataset.resources ? dataset.resources.length : 0}
                  </TableCell>
                  <TableCell align="center">
                    <Button
                      size="small"
                      color="primary"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDatasetSelect(dataset);
                      }}
                    >
                      {ui.view_details}
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
        
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
          <Pagination 
            count={totalPages} 
            page={page} 
            onChange={handlePageChange} 
            color="primary" 
          />
        </Box>
      </>
    );
  };

  // Render the dataset list for the alternative API
  const renderAlternativeDatasetList = () => {
    if (!loadedAltDatasets) {
      return (
        <Box sx={{ textAlign: 'center', mt: 2 }}>
          <Button
            variant="contained"
            color="primary"
            onClick={loadAltDatasets}
            disabled={loading}
            startIcon={loading ? <CircularProgress size={20} /> : null}
          >
            {ui.load_datasets}
          </Button>
          <Typography variant="caption" display="block" sx={{ mt: 1 }}>
            {loading ? ui.loading_alt_datasets : ''}
          </Typography>
        </Box>
      );
    }

    if (loading) {
      return (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
          <CircularProgress />
          <Typography variant="body2" sx={{ ml: 2 }}>
            {ui.loading_alt_datasets}
          </Typography>
        </Box>
      );
    }

    if (error) {
      return (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      );
    }

    if (filteredAltDatasets.length === 0) {
      return (
        <Typography variant="body2" sx={{ p: 2, textAlign: 'center' }}>
          {ui.no_alt_datasets}
        </Typography>
      );
    }

    // Calculate pagination
    const startIndex = (page - 1) * rowsPerPage;
    const endIndex = startIndex + rowsPerPage;
    const paginatedDatasets = filteredAltDatasets.slice(startIndex, endIndex);
    const totalPages = Math.ceil(filteredAltDatasets.length / rowsPerPage);

    return (
      <>
        <Box sx={{ mb: 2 }}>
          <TextField
            fullWidth
            label={ui.dataset_filter}
            value={altDatasetFilter}
            onChange={handleAltDatasetFilterChange}
            variant="outlined"
            size="small"
          />
        </Box>
        
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>{ui.dataset_id}</TableCell>
                <TableCell>{ui.dataset_title}</TableCell>
                <TableCell>{ui.dataset_type}</TableCell>
                <TableCell align="center">{ui.dataset_actions}</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {paginatedDatasets.map((dataset) => (
                <TableRow 
                  key={dataset.id}
                  hover
                  selected={selectedAltDataset?.id === dataset.id}
                  onClick={() => handleAltDatasetSelect(dataset)}
                  sx={{ cursor: 'pointer' }}
                >
                  <TableCell>{dataset.id}</TableCell>
                  <TableCell>{dataset.title}</TableCell>
                  <TableCell>{dataset.type}</TableCell>
                  <TableCell align="center">
                    <Button
                      size="small"
                      color="primary"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleAltDatasetSelect(dataset);
                      }}
                    >
                      {ui.view_details}
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
        
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
          <Pagination 
            count={totalPages} 
            page={page} 
            onChange={handlePageChange} 
            color="primary" 
          />
        </Box>
      </>
    );
  };

  // Render the dataset details for the primary API
  const renderPrimaryDatasetDetails = () => {
    if (!selectedDataset) return null;

    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            {selectedDataset.title}
          </Typography>
          
          <Typography variant="body2" paragraph>
            {selectedDataset.notes}
          </Typography>

          <Box>
            <Typography variant="subtitle2" gutterBottom>
              {ui.dataset_details}:
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
              {selectedDataset.tags && selectedDataset.tags.map(tag => (
                <Chip key={tag.id} label={tag.display_name} size="small" color="primary" variant="outlined" />
              ))}
            </Box>
          </Box>
          
          <Tabs value={tabValue} onChange={handleTabChange} sx={{ mb: 2 }}>
            <Tab label={ui.raw_data} />
            {csvData && <Tab label={ui.visualization} />}
          </Tabs>
          
          <TabPanel value={tabValue} index={0}>
            {resourceLoading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                <CircularProgress />
                <Typography variant="body2" sx={{ ml: 2 }}>
                  {ui.loading_resources}
                </Typography>
              </Box>
            ) : resources.length === 0 ? (
              <Typography variant="body2" sx={{ p: 2 }}>
                {ui.no_resources}
              </Typography>
            ) : (
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>{ui.resource_name}</TableCell>
                      <TableCell>{ui.resource_format}</TableCell>
                      <TableCell align="right">{ui.resource_size}</TableCell>
                      <TableCell align="center">{ui.dataset_actions}</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {resources.map((resource) => (
                      <TableRow key={resource.id}>
                        <TableCell>{resource.name}</TableCell>
                        <TableCell>{resource.format}</TableCell>
                        <TableCell align="right">{resource.size ? `${Math.round(resource.size / 1024)} KB` : 'N/A'}</TableCell>
                        <TableCell align="center">
                          <Button
                            size="small"
                            variant="outlined"
                            onClick={() => handleResourceSelect(resource)}
                            sx={{ mr: 1 }}
                          >
                            {ui.view_resource}
                          </Button>
                          <IconButton
                            color="primary"
                            size="small"
                            onClick={() => downloadResource(resource)}
                          >
                            <DownloadIcon />
                          </IconButton>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            )}
          </TabPanel>
          
          <TabPanel value={tabValue} index={1}>
            {csvData && (
              <TableContainer component={Paper} sx={{ mt: 2 }}>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      {csvData.columns.map((column, index) => (
                        <TableCell key={index}>{column}</TableCell>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {csvData.rows.map((row, rowIndex) => (
                      <TableRow key={rowIndex}>
                        {row.map((cell, cellIndex) => (
                          <TableCell key={cellIndex}>{cell}</TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            )}
          </TabPanel>
        </CardContent>
        <CardActions>
          <Button 
            size="small" 
            color="primary"
            onClick={() => {
              setSelectedDataset(null);
              setResources([]);
              setCsvData(null);
            }}
          >
            Back to List
          </Button>
        </CardActions>
      </Card>
    );
  };

  // Render the dataset details for the alternative API
  const renderAlternativeDatasetDetails = () => {
    if (!selectedAltDataset) return null;
    if (resourceLoading) {
      return (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
          <CircularProgress />
          <Typography variant="body2" sx={{ ml: 2 }}>
            {ui.loading_resources}
          </Typography>
        </Box>
      );
    }

    if (!altDatasetDetails) {
      return (
        <Alert severity="error" sx={{ mt: 2 }}>
          Failed to load dataset details.
        </Alert>
      );
    }

    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            {altDatasetDetails.title || selectedAltDataset.title}
          </Typography>
          
          <Typography variant="body2" paragraph>
            {altDatasetDetails.description || 'No description available.'}
          </Typography>

          <Typography variant="subtitle2" gutterBottom>
            {ui.dataset_details}:
          </Typography>
          <TableContainer component={Paper} sx={{ mb: 2 }}>
            <Table size="small">
              <TableBody>
                <TableRow>
                  <TableCell component="th" scope="row" sx={{ fontWeight: 'bold' }}>
                    ID
                  </TableCell>
                  <TableCell>{altDatasetDetails.id}</TableCell>
                </TableRow>
                {altDatasetDetails.type && (
                  <TableRow>
                    <TableCell component="th" scope="row" sx={{ fontWeight: 'bold' }}>
                      Type
                    </TableCell>
                    <TableCell>{altDatasetDetails.type}</TableCell>
                  </TableRow>
                )}
                {altDatasetDetails.resources && (
                  <TableRow>
                    <TableCell component="th" scope="row" sx={{ fontWeight: 'bold' }}>
                      Resources
                    </TableCell>
                    <TableCell>{altDatasetDetails.resources.length}</TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>

          {altDatasetDetails.resources && altDatasetDetails.resources.length > 0 ? (
            <>
              <Typography variant="subtitle2" gutterBottom>
                Resources:
              </Typography>
              <TableContainer component={Paper}>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Name</TableCell>
                      <TableCell>Format</TableCell>
                      <TableCell align="right">Action</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {altDatasetDetails.resources.map((resource, index) => (
                      <TableRow key={index}>
                        <TableCell>{resource.name || `Resource ${index + 1}`}</TableCell>
                        <TableCell>{resource.format || 'Unknown'}</TableCell>
                        <TableCell align="right">
                          {resource.url && (
                            <Button
                              size="small"
                              color="primary"
                              variant="outlined"
                              endIcon={<DownloadIcon />}
                              onClick={() => window.open(resource.url, '_blank')}
                            >
                              {ui.download_resource}
                            </Button>
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </>
          ) : (
            <Typography variant="body2" color="textSecondary">
              No resources available for this dataset.
            </Typography>
          )}
        </CardContent>
        <CardActions>
          <Button 
            size="small" 
            color="primary"
            onClick={() => {
              setSelectedAltDataset(null);
              setAltDatasetDetails(null);
            }}
          >
            Back to List
          </Button>
          {altDatasetDetails.url && (
            <Button
              size="small"
              color="primary"
              endIcon={<InfoIcon />}
              onClick={() => window.open(altDatasetDetails.url, '_blank')}
            >
              View on BDAP
            </Button>
          )}
        </CardActions>
      </Card>
    );
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        {ui.title}
      </Typography>
      <Typography variant="body1" paragraph>
        {ui.description}
      </Typography>

      {/* API Selection */}
      <Paper elevation={1} sx={{ p: 2, mb: 3 }}>
        <FormControl component="fieldset">
          <FormLabel component="legend">{ui.api_selection}</FormLabel>
          <RadioGroup
            row
            name="api-type"
            value={apiType}
            onChange={handleApiTypeChange}
          >
            <FormControlLabel
              value="primary"
              control={<Radio />}
              label={ui.primary_api}
            />
            <FormControlLabel
              value="alternative"
              control={<Radio />}
              label={ui.alternative_api}
            />
          </RadioGroup>
        </FormControl>
      </Paper>

      {/* Dataset search for primary API */}
      {apiType === 'primary' && !selectedDataset && (
        <Paper elevation={1} sx={{ p: 2, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            {ui.search_datasets}
          </Typography>
          <Box sx={{ display: 'flex', mb: 2 }}>
            <TextField
              fullWidth
              label={ui.search_placeholder}
              variant="outlined"
              value={searchQuery}
              onChange={handleSearchQueryChange}
              sx={{ mr: 1 }}
            />
            <Button
              variant="contained"
              color="primary"
              onClick={handleSearch}
              disabled={loading}
              startIcon={<SearchIcon />}
            >
              {ui.search_button}
            </Button>
          </Box>
          {renderPrimaryDatasetList()}
        </Paper>
      )}

      {/* Dataset list for alternative API */}
      {apiType === 'alternative' && !selectedAltDataset && (
        <Paper elevation={1} sx={{ p: 2, mb: 3 }}>
          {renderAlternativeDatasetList()}
        </Paper>
      )}

      {/* Dataset details for primary API */}
      {apiType === 'primary' && selectedDataset && (
        <Paper elevation={1} sx={{ p: 2 }}>
          {renderPrimaryDatasetDetails()}
        </Paper>
      )}

      {/* Dataset details for alternative API */}
      {apiType === 'alternative' && selectedAltDataset && (
        <Paper elevation={1} sx={{ p: 2 }}>
          {renderAlternativeDatasetDetails()}
        </Paper>
      )}

      {/* External resources */}
      <Paper elevation={1} sx={{ p: 2, mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          {ui.external_resources}
        </Typography>
        <Typography variant="body2" paragraph>
          {ui.external_resources_info}
        </Typography>
        <Link 
          href="https://bdap-opendata.rgs.mef.gov.it/metastore" 
          target="_blank" 
          rel="noopener noreferrer"
        >
          {ui.bdap_opendata}
        </Link>
      </Paper>
    </Box>
  );
};

// TabPanel component for the tabs
const TabPanel = (props) => {
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
        <Box>
          {children}
        </Box>
      )}
    </div>
  );
};

export default FinancialDataTab;
