export function dataSourceForId(schema, sourceId) {
  const id = String(sourceId || "dataset");
  const sources = schema?.data_sources || [];
  const source = sources.find((item) => item.id === id);
  if (source) return source;
  return id === "dataset" ? { id: "dataset", columns: schema?.columns || [] } : null;
}

export function currentDataSource(schema, sourceId) {
  return dataSourceForId(schema, sourceId || "dataset") || dataSourceForId(schema, "dataset");
}

export function sourceColumns(schema, sourceId) {
  return currentDataSource(schema, sourceId)?.columns || schema?.columns || [];
}

export function dataSourceColumns(schema, sourceId) {
  return dataSourceForId(schema, sourceId)?.columns || [];
}

export function dataSourceHasColumn(schema, sourceId, columnName) {
  const name = String(columnName || "");
  return Boolean(name && dataSourceColumns(schema, sourceId).some((column) => column.name === name));
}

export function toolEnabled(schema, id) {
  return Boolean((schema?.tools || []).some((tool) => tool.id === id));
}

export function isModelTool(tool) {
  return tool === "glm" || tool === "gbm";
}

export function isModelPredictionColumn(column) {
  return ["gbm_prediction", "gbm_prediction_rate", "gbm_tabulated_prediction", "glm_prediction", "glm_prediction_rate", "glm_tabulated_prediction"].includes(String(column?.name || ""));
}

export function preferredStartupSource(availableSources, requestedSource) {
  if (availableSources.some((source) => source.id === requestedSource)) return requestedSource;
  const activePredictionSource = availableSources.find((source) => ["glm_predictions", "gbm_predictions"].includes(source.kind) && source.active);
  if (activePredictionSource) return activePredictionSource.id;
  const predictionSource = availableSources.find((source) => ["glm_predictions", "gbm_predictions"].includes(source.kind));
  return predictionSource?.id || "dataset";
}
