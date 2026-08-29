let adapterPromise = null;

export function loadMapLibreAdapter() {
  if (!adapterPromise) {
    adapterPromise = import("../vendor/maplibre-gl/maplibre-gl.mjs")
      .then((maplibregl) => createMapLibreAdapter(maplibregl));
  }
  return adapterPromise;
}

export function createMapLibreAdapter(maplibregl) {
  let nextObjectId = 1;

  function emptyMapStyle() {
    return {
      version: 8,
      sources: {},
      layers: [{
        id: "lucidum-background",
        type: "background",
        paint: { "background-color": "rgba(0, 0, 0, 0)" },
      }],
    };
  }

  function objectId(prefix) {
    const id = `${prefix}-${nextObjectId}`;
    nextObjectId += 1;
    return id;
  }

  function latLng(value, longitude) {
    if (Number.isFinite(Number(value)) && Number.isFinite(Number(longitude))) {
      return { lat: Number(value), lng: Number(longitude) };
    }
    if (Array.isArray(value)) return { lat: Number(value[0]), lng: Number(value[1]) };
    return { lat: Number(value?.lat), lng: Number(value?.lng ?? value?.lon) };
  }

  function lngLatLike(value) {
    const point = latLng(value);
    return [point.lng, point.lat];
  }

  class Bounds {
    constructor(value = null) {
      this.raw = new maplibregl.LngLatBounds();
      if (value instanceof Bounds) {
        if (value.isValid()) this.raw.extend(value.raw);
      } else if (value instanceof maplibregl.LngLatBounds) {
        if (!value.isEmpty()) this.raw.extend(value);
      } else if (Array.isArray(value) && value.length >= 2) {
        this.extend(value[0]);
        this.extend(value[1]);
      }
    }

    extend(value) {
      if (value instanceof Bounds) {
        if (value.isValid()) this.raw.extend(value.raw);
        return this;
      }
      if (value instanceof maplibregl.LngLatBounds) {
        if (!value.isEmpty()) this.raw.extend(value);
        return this;
      }
      const point = latLng(value);
      if (Number.isFinite(point.lat) && Number.isFinite(point.lng)) {
        this.raw.extend([point.lng, point.lat]);
      }
      return this;
    }

    isValid() {
      return !this.raw.isEmpty();
    }

    getSouth() {
      return this.raw.getSouth();
    }

    getWest() {
      return this.raw.getWest();
    }

    getNorth() {
      return this.raw.getNorth();
    }

    getEast() {
      return this.raw.getEast();
    }

    getCenter() {
      const center = this.raw.getCenter();
      return { lat: center.lat, lng: center.lng };
    }

    getSouthWest() {
      return { lat: this.getSouth(), lng: this.getWest() };
    }

    getNorthWest() {
      return { lat: this.getNorth(), lng: this.getWest() };
    }

    getNorthEast() {
      return { lat: this.getNorth(), lng: this.getEast() };
    }

    getSouthEast() {
      return { lat: this.getSouth(), lng: this.getEast() };
    }
  }

  function extendBoundsFromCoordinates(bounds, coordinates) {
    if (!Array.isArray(coordinates)) return;
    if (
      coordinates.length >= 2
      && Number.isFinite(Number(coordinates[0]))
      && Number.isFinite(Number(coordinates[1]))
    ) {
      bounds.extend([Number(coordinates[1]), Number(coordinates[0])]);
      return;
    }
    coordinates.forEach((item) => extendBoundsFromCoordinates(bounds, item));
  }

  function featureBounds(feature) {
    const bounds = new Bounds();
    const geometry = feature?.geometry;
    if (geometry?.type === "GeometryCollection") {
      (geometry.geometries || []).forEach((item) => extendBoundsFromCoordinates(bounds, item.coordinates));
    } else {
      extendBoundsFromCoordinates(bounds, geometry?.coordinates);
    }
    return bounds;
  }

  function normalizePadding(padding) {
    if (Array.isArray(padding)) {
      const horizontal = Number(padding[0]) || 0;
      const vertical = Number(padding[1] ?? padding[0]) || 0;
      return { top: vertical, right: horizontal, bottom: vertical, left: horizontal };
    }
    return padding;
  }

  function resolveContent(content, context) {
    return typeof content === "function" ? content.call(context) : String(content ?? "");
  }

  class PopupLayer {
    constructor(options = {}, tooltip = false) {
      this.options = { ...options };
      this.tooltip = tooltip;
      this.map = null;
      this.raw = null;
      this.position = null;
      this.content = "";
      this._closeHandlers = [];
    }

    ensureRaw() {
      if (this.raw) return this.raw;
      const maxWidth = this.options.maxWidth;
      this.raw = new maplibregl.Popup({
        closeButton: this.tooltip ? false : this.options.closeButton !== false,
        closeOnClick: this.tooltip ? false : this.options.closeOnClick !== false,
        focusAfterOpen: false,
        className: this.tooltip ? "maplibre-tooltip" : "",
        maxWidth: Number.isFinite(Number(maxWidth)) ? `${Number(maxWidth)}px` : (maxWidth || "240px"),
      });
      this.raw.on("close", () => {
        const previousMap = this.map;
        this.map = null;
        if (previousMap?._popup === this) previousMap._popup = null;
        previousMap?.fire("popupclose", { popup: this });
        this._closeHandlers.forEach((handler) => handler({ popup: this }));
      });
      return this.raw;
    }

    setLatLng(value) {
      this.position = latLng(value);
      this.ensureRaw().setLngLat(lngLatLike(this.position));
      return this;
    }

    getLatLng() {
      return this.position;
    }

    setContent(content) {
      this.content = content;
      this.ensureRaw().setHTML(resolveContent(content, this));
      return this;
    }

    setText(content) {
      this.content = String(content ?? "");
      this.ensureRaw().setText(this.content);
      return this;
    }

    getContent() {
      return this.content;
    }

    addTo(map) {
      const target = map instanceof MapFacade ? map : map?.map;
      if (!target || !this.position) return this;
      if (!this.tooltip) target.closePopup();
      this.map = target;
      const raw = this.ensureRaw();
      raw.setLngLat(lngLatLike(this.position));
      raw.setHTML(resolveContent(this.content, this));
      raw.addTo(target.raw);
      if (!this.tooltip) target._popup = this;
      return this;
    }

    openOn(map) {
      return this.addTo(map);
    }

    remove() {
      this.raw?.remove();
      return this;
    }

    update() {
      this.raw?._update?.();
      return this;
    }

    isOpen() {
      return Boolean(this.raw?.isOpen?.());
    }

    on(type, handler) {
      if (type === "close" && typeof handler === "function") this._closeHandlers.push(handler);
      return this;
    }

    getElement() {
      return this.raw?.getElement?.() || null;
    }
  }

  class LayerBase {
    constructor(...args) {
      this._lucidumLayerId = objectId("layer");
      this.map = null;
      if (typeof this.initialize === "function") this.initialize(...args);
    }

    addTo(target) {
      if (target instanceof LayerGroup) {
        target.addLayer(this);
        return this;
      }
      if (!(target instanceof MapFacade)) return this;
      if (this.map === target) return this;
      this.map = target;
      target._registerLayer(this);
      this.onAdd?.(target);
      target._markRenderPending();
      return this;
    }

    remove() {
      const previousMap = this.map;
      if (!previousMap) return this;
      this.onRemove?.(previousMap);
      previousMap._unregisterLayer(this);
      previousMap._markRenderPending();
      this.map = null;
      return this;
    }

    static extend(definition = {}) {
      class ExtendedLayer extends this {}
      Object.assign(ExtendedLayer.prototype, definition);
      return ExtendedLayer;
    }
  }

  class MapFacade {
    constructor(container, options = {}) {
      this.options = { ...options };
      this._container = typeof container === "string" ? document.getElementById(container) : container;
      this._container.tabIndex = 0;
      this._layers = {};
      this._events = [];
      this._customEvents = new Map();
      this._popup = null;
      this._pendingSourceIds = new Set();
      this._renderPending = false;
      this._styleGeneration = 0;
      this._styleForegroundLayerIds = [];
      this._usesExternalStyle = false;
      this.raw = new maplibregl.Map({
        container: this._container,
        style: emptyMapStyle(),
        attributionControl: true,
        center: [-3.2, 54.5],
        zoom: 5,
        dragRotate: false,
        pitchWithRotate: false,
        touchPitch: false,
        maplibreLogo: true,
      });
      this._readyPromise = new Promise((resolve, reject) => {
        this.raw.once("load", () => resolve(this));
        this.raw.once("error", (event) => {
          if (!this.raw.loaded()) reject(event?.error || new Error("MapLibre could not initialise WebGL2."));
        });
      });
      this._styleReadyPromise = this._readyPromise;
      this._container._lucidumMap = this;
      this._container._lucidumMapLibre = this.raw;
    }

    whenReady() {
      return this._readyPromise;
    }

    whenStyleReady() {
      return this._styleReadyPromise;
    }

    usesExternalStyle() {
      return this._usesExternalStyle;
    }

    getStyleForegroundLayerIds() {
      return [...this._styleForegroundLayerIds];
    }

    bringStyleForegroundToFront() {
      this._styleForegroundLayerIds.forEach((layerId) => {
        if (this.raw.getLayer(layerId)) this.raw.moveLayer(layerId);
      });
      return this;
    }

    replaceStyle(style = null, { foregroundLayerPredicate = null, timeoutMs = 15000 } = {}) {
      const generation = ++this._styleGeneration;
      const nextStyle = style || emptyMapStyle();
      this._usesExternalStyle = Boolean(style);
      this._styleForegroundLayerIds = [];
      Object.values(this._layers).forEach((layer) => layer.beforeStyleReplace?.(this));

      const replacement = this.whenReady().then(() => new Promise((resolve, reject) => {
        let settled = false;
        let timer = null;
        const cleanup = () => {
          if (timer !== null) window.clearTimeout(timer);
          this.raw.off("style.load", handleStyleLoad);
          this.raw.off("error", handleError);
        };
        const finish = (value, error = null) => {
          if (settled) return;
          settled = true;
          cleanup();
          if (error) reject(error);
          else resolve(value);
        };
        const handleStyleLoad = () => {
          if (generation !== this._styleGeneration) {
            finish(false);
            return;
          }
          try {
            const styleLayers = this.raw.getStyle()?.layers || [];
            this._styleForegroundLayerIds = typeof foregroundLayerPredicate === "function"
              ? styleLayers.filter((layer) => foregroundLayerPredicate(layer)).map((layer) => layer.id)
              : [];
            Object.values(this._layers).forEach((layer) => layer.restoreStyleResources?.(this));
            this.bringStyleForegroundToFront();
            this._markRenderPending();
            finish(true);
          } catch (error) {
            finish(false, error);
          }
        };
        const handleError = (event) => {
          if (generation !== this._styleGeneration) {
            finish(false);
            return;
          }
          if (event?.sourceId) return;
          finish(false, event?.error || new Error("MapLibre could not load the requested style."));
        };

        this.raw.on("style.load", handleStyleLoad);
        this.raw.on("error", handleError);
        timer = window.setTimeout(() => {
          if (generation !== this._styleGeneration) finish(false);
          else finish(false, new Error("MapLibre style loading timed out."));
        }, Math.max(1000, Number(timeoutMs) || 15000));
        try {
          this.raw.setStyle(nextStyle, { diff: false });
        } catch (error) {
          finish(false, error);
        }
      }));
      this._styleReadyPromise = replacement;
      return replacement;
    }

    _registerLayer(layer) {
      this._layers[layer._lucidumLayerId || objectId("layer")] = layer;
    }

    _unregisterLayer(layer) {
      const layerId = layer?._lucidumLayerId;
      if (layerId && this._layers[layerId] === layer) {
        delete this._layers[layerId];
        return;
      }
      Object.entries(this._layers).forEach(([id, candidate]) => {
        if (candidate === layer) delete this._layers[id];
      });
    }

    _registerFeatureLayer(layer) {
      this._registerLayer(layer);
    }

    _markSourcePending(sourceId) {
      if (sourceId) this._pendingSourceIds.add(sourceId);
      this._markRenderPending();
    }

    _markRenderPending() {
      this._renderPending = true;
    }

    async whenRenderComplete(timeoutMs = 3000) {
      await this.whenReady();
      const sourceIds = Array.from(this._pendingSourceIds);
      this._pendingSourceIds.clear();
      const waitForSource = (sourceId) => new Promise((resolve) => {
        if (!this.raw.getSource(sourceId)) {
          resolve();
          return;
        }
        try {
          if (this.raw.isSourceLoaded(sourceId)) {
            resolve();
            return;
          }
        } catch (_) {
        }
        let timer = null;
        const finish = () => {
          if (timer !== null) window.clearTimeout(timer);
          this.raw.off("sourcedata", handleSourceData);
          resolve();
        };
        const handleSourceData = (event) => {
          if (event.sourceId === sourceId && event.isSourceLoaded) finish();
        };
        this.raw.on("sourcedata", handleSourceData);
        timer = window.setTimeout(finish, timeoutMs);
      });
      await Promise.all(sourceIds.map(waitForSource));
      if (this._renderPending) {
        await new Promise((resolve) => {
          let settled = false;
          const finish = () => {
            if (settled) return;
            settled = true;
            window.clearTimeout(timer);
            resolve();
          };
          const timer = window.setTimeout(finish, Math.min(timeoutMs, 500));
          this.raw.once("render", finish);
          this.raw.triggerRepaint();
        });
      }
      this._renderPending = false;
      await new Promise((resolve) => requestAnimationFrame(() => resolve()));
    }

    setView(center, zoom, options = {}) {
      const camera = { center: lngLatLike(center), zoom: Number(zoom) - 1 };
      const bearing = Number(options.bearing);
      if (Number.isFinite(bearing)) camera.bearing = bearing;
      this.raw.jumpTo(camera);
      return this;
    }

    panTo(center) {
      this.raw.jumpTo({ center: lngLatLike(center) });
      return this;
    }

    getCenter() {
      const center = this.raw.getCenter();
      return { lat: center.lat, lng: center.lng };
    }

    getZoom() {
      return this.raw.getZoom() + 1;
    }

    getBearing() {
      return this.raw.getBearing();
    }

    setBearing(bearing) {
      const nextBearing = Number(bearing);
      if (Number.isFinite(nextBearing)) this.raw.setBearing(nextBearing);
      return this;
    }

    isZooming() {
      return Boolean(this.raw.isZooming?.());
    }

    setZoom(zoom) {
      this.raw.jumpTo({ zoom: Number(zoom) - 1 });
      return this;
    }

    getBounds() {
      return new Bounds(this.raw.getBounds());
    }

    getBoundsZoom(bounds, options = {}) {
      if (!(bounds instanceof Bounds) || !bounds.isValid()) return null;
      const nextOptions = {
        ...options,
        padding: normalizePadding(options.padding),
      };
      delete nextOptions.animate;
      delete nextOptions.duration;
      if (Number.isFinite(Number(nextOptions.maxZoom))) nextOptions.maxZoom = Number(nextOptions.maxZoom) - 1;
      try {
        const camera = this.raw.cameraForBounds(bounds.raw, nextOptions);
        return Number.isFinite(Number(camera?.zoom)) ? Number(camera.zoom) + 1 : null;
      } catch (_) {
        return null;
      }
    }

    fitBounds(bounds, options = {}) {
      if (!(bounds instanceof Bounds) || !bounds.isValid()) return this;
      const nextOptions = {
        ...options,
        padding: normalizePadding(options.padding),
        duration: options.animate === false ? 0 : options.duration,
      };
      delete nextOptions.animate;
      if (Number.isFinite(Number(nextOptions.maxZoom))) nextOptions.maxZoom = Number(nextOptions.maxZoom) - 1;
      this.raw.fitBounds(bounds.raw, nextOptions);
      return this;
    }

    invalidateSize() {
      this.raw.resize();
      return this;
    }

    getContainer() {
      return this._container;
    }

    getSize() {
      return { x: this._container.clientWidth, y: this._container.clientHeight };
    }

    containerPointToLatLng(point) {
      const result = this.raw.unproject(Array.isArray(point) ? point : [point.x, point.y]);
      return { lat: result.lat, lng: result.lng };
    }

    latLngToContainerPoint(value) {
      return this.raw.project(lngLatLike(value));
    }

    getPixelBounds() {
      const size = this.getSize();
      const zoom = this.getZoom();
      const worldSize = 256 * (2 ** zoom);
      const center = this.getCenter();
      const sinLatitude = Math.sin((Math.max(-85.0511287798, Math.min(85.0511287798, center.lat)) * Math.PI) / 180);
      const centerX = ((center.lng + 180) / 360) * worldSize;
      const centerY = (0.5 - (Math.log((1 + sinLatitude) / (1 - sinLatitude)) / (4 * Math.PI))) * worldSize;
      return {
        min: { x: centerX - (size.x / 2), y: centerY - (size.y / 2) },
        max: { x: centerX + (size.x / 2), y: centerY + (size.y / 2) },
      };
    }

    hasLayer(layer) {
      if (layer instanceof PopupLayer) return layer.isOpen();
      return Object.values(this._layers).includes(layer);
    }

    eachLayer(callback) {
      Object.values(this._layers).forEach(callback);
      return this;
    }

    removeLayer(layer) {
      layer?.remove?.();
      return this;
    }

    closePopup() {
      this._popup?.remove();
      return this;
    }

    on(types, handler, context = null) {
      String(types || "").split(/\s+/).filter(Boolean).forEach((type) => {
        if (type === "popupclose") {
          const handlers = this._customEvents.get(type) || [];
          handlers.push({ handler, context });
          this._customEvents.set(type, handlers);
          return;
        }
        const bound = (event) => handler.call(context || this, {
          ...event,
          containerPoint: event?.point || event?.containerPoint,
          latlng: event?.lngLat ? { lat: event.lngLat.lat, lng: event.lngLat.lng } : event?.latlng,
        });
        this._events.push({ type, handler, context, bound });
        this.raw.on(type, bound);
      });
      return this;
    }

    off(types, handler, context = null) {
      const requestedTypes = new Set(String(types || "").split(/\s+/).filter(Boolean));
      this._events = this._events.filter((entry) => {
        const matches = requestedTypes.has(entry.type) && entry.handler === handler && entry.context === context;
        if (matches) this.raw.off(entry.type, entry.bound);
        return !matches;
      });
      requestedTypes.forEach((type) => {
        if (type !== "popupclose") return;
        const handlers = this._customEvents.get(type) || [];
        this._customEvents.set(type, handlers.filter((entry) => entry.handler !== handler || entry.context !== context));
      });
      return this;
    }

    fire(type, payload = {}) {
      const customHandlers = this._customEvents.get(type) || [];
      customHandlers.forEach(({ handler, context }) => handler.call(context || this, payload));
      this._events
        .filter((entry) => entry.type === type)
        .forEach((entry) => entry.handler.call(entry.context || this, payload));
      return this;
    }
  }

  class RasterLayer extends LayerBase {
    initialize(url, options = {}) {
      this.url = url;
      this._url = url;
      this.options = { ...options };
      this.sourceId = objectId("lucidum-raster-source");
      this.layerId = objectId("lucidum-raster-layer");
      this.layerPosition = "";
    }

    tileUrls() {
      const values = this.url.includes("{s}") ? ["a", "b", "c", "d"] : [""];
      const retina = window.devicePixelRatio > 1 ? "@2x" : "";
      return values.map((subdomain) => this.url.replace("{s}", subdomain).replace("{r}", retina));
    }

    restoreStyleResources(map = this.map) {
      if (!map || this.map !== map || map.raw.getSource(this.sourceId)) return this;
      map.raw.addSource(this.sourceId, {
        type: "raster",
        tiles: this.tileUrls(),
        tileSize: 256,
        maxzoom: Number(this.options.maxZoom) || 19,
        attribution: this.options.attribution || "",
      });
      const firstOverlay = Object.values(map._layers)
        .map((candidate) => candidate !== this && (candidate.fillLayerId || (candidate.canvas && candidate.layerId)))
        .find((layerId) => layerId && map.raw.getLayer(layerId));
      map.raw.addLayer({
        id: this.layerId,
        type: "raster",
        source: this.sourceId,
      }, firstOverlay);
      this.applyLayerPosition();
      map._markRenderPending();
      return this;
    }

    onAdd(map) {
      const add = () => this.restoreStyleResources(map);
      if (map.raw.isStyleLoaded?.()) add();
      else map.whenStyleReady().then(add).catch(() => {});
    }

    onRemove(map) {
      if (map.raw.getLayer(this.layerId)) map.raw.removeLayer(this.layerId);
      if (map.raw.getSource(this.sourceId)) map.raw.removeSource(this.sourceId);
    }

    applyLayerPosition() {
      if (!this.map?.raw.getLayer(this.layerId)) return this;
      if (this.layerPosition === "front") {
        this.map.raw.moveLayer(this.layerId);
        return this;
      }
      if (this.layerPosition !== "back") return this;
      const firstOverlay = Object.values(this.map._layers)
        .map((candidate) => candidate !== this && (candidate.fillLayerId || (candidate.canvas && candidate.layerId)))
        .find((layerId) => layerId && this.map.raw.getLayer(layerId));
      if (firstOverlay) this.map.raw.moveLayer(this.layerId, firstOverlay);
      return this;
    }

    bringToBack() {
      this.layerPosition = "back";
      return this.applyLayerPosition();
    }

    bringToFront() {
      this.layerPosition = "front";
      return this.applyLayerPosition();
    }
  }

  class CanvasLayer extends LayerBase {
    initialize(canvas, coordinates, options = {}) {
      this.canvas = canvas;
      this.coordinates = coordinates;
      this.options = { ...options };
      this.sourceId = objectId("lucidum-canvas-source");
      this.layerId = objectId("lucidum-canvas-layer");
      this.visible = true;
      this.renderListeners = new Set();
    }

    beforeStyleReplace(map) {
      this.renderListeners.forEach((listener) => map.raw.off("render", listener));
      this.renderListeners.clear();
    }

    restoreStyleResources(map = this.map) {
      if (!map || this.map !== map || map.raw.getSource(this.sourceId)) return this;
      map.raw.addSource(this.sourceId, {
        type: "canvas",
        canvas: this.canvas,
        coordinates: this.coordinates,
        animate: false,
      });
      map.raw.addLayer({
        id: this.layerId,
        type: "raster",
        source: this.sourceId,
        layout: { visibility: this.visible ? "visible" : "none" },
        paint: {
          "raster-fade-duration": 0,
          "raster-opacity": this.visible ? 1 : 0,
        },
      });
      map._markRenderPending();
      return this;
    }

    onAdd(map) {
      const add = () => this.restoreStyleResources(map);
      if (map.raw.isStyleLoaded?.()) add();
      else map.whenStyleReady().then(add).catch(() => {});
    }

    onRemove(map) {
      this.renderListeners.forEach((listener) => map.raw.off("render", listener));
      this.renderListeners.clear();
      if (map.raw.getLayer(this.layerId)) map.raw.removeLayer(this.layerId);
      if (map.raw.getSource(this.sourceId)) map.raw.removeSource(this.sourceId);
    }

    setCoordinates(coordinates) {
      this.coordinates = coordinates;
      this.map?.raw.getSource(this.sourceId)?.setCoordinates?.(coordinates);
      return this;
    }

    setVisible(visible) {
      this.visible = Boolean(visible);
      if (this.map?.raw.getLayer(this.layerId)) {
        this.map.raw.setLayoutProperty(
          this.layerId,
          "visibility",
          this.visible ? "visible" : "none",
        );
        this.map.raw.setPaintProperty(this.layerId, "raster-opacity", this.visible ? 1 : 0);
      }
      return this;
    }

    refresh({ afterRender = null } = {}) {
      const source = this.map?.raw.getSource(this.sourceId);
      if (!source) return this;
      source.play?.();
      this.map._markRenderPending();
      if (typeof afterRender === "function") {
        const map = this.map;
        const listener = () => {
          map.raw.off("render", listener);
          this.renderListeners.delete(listener);
          if (map.raw.getSource(this.sourceId) !== source) return;
          source.pause?.();
          afterRender();
        };
        this.renderListeners.add(listener);
        map.raw.on("render", listener);
      } else {
        requestAnimationFrame(() => source.pause?.());
      }
      this.map.raw.triggerRepaint();
      return this;
    }

    bringToFront() {
      if (this.map?.raw.getLayer(this.layerId)) this.map.raw.moveLayer(this.layerId);
      return this;
    }
  }

  class FeatureLayer {
    constructor(parent, feature, id) {
      this.parent = parent;
      this.feature = feature;
      this.featureId = id;
      this._lucidumLayerId = objectId("feature");
      this._bounds = featureBounds(feature);
      this._tooltipContent = null;
      this._popupContent = null;
      this._popupOptions = {};
      this._popup = null;
      this.options = {};
    }

    bindTooltip(content, options = {}) {
      this._tooltipContent = content;
      this._tooltipOptions = options;
      return this;
    }

    bindPopup(content, options = {}) {
      this._popupContent = content;
      this._popupOptions = options;
      return this;
    }

    getPopup() {
      return this._popup;
    }

    getBounds() {
      return this._bounds;
    }

    getCenter() {
      return this._bounds.getCenter();
    }

    bringToFront() {
      return this;
    }

    openPopup() {
      return this.parent.openFeaturePopup(this);
    }
  }

  class GeoJsonLayer extends LayerBase {
    initialize(geoJson, options = {}) {
      this.options = { ...options };
      this.sourceId = objectId("lucidum-geojson-source");
      this.fillLayerId = objectId("lucidum-geojson-fill");
      this.lineLayerId = objectId("lucidum-geojson-line");
      this.bounds = new Bounds();
      this.features = (geoJson?.features || []).map((feature, index) => {
        const id = String(index);
        const sourceFeature = {
          ...feature,
          id,
          properties: { ...(feature.properties || {}), __lucidum_feature_id: id },
        };
        const layer = new FeatureLayer(this, sourceFeature, id);
        if (layer.getBounds().isValid()) this.bounds.extend(layer.getBounds());
        this.options.onEachFeature?.(sourceFeature, layer);
        return layer;
      });
      this.geoJson = {
        type: "FeatureCollection",
        features: this.features.map((layer) => layer.feature),
      };
      this.tooltip = new PopupLayer({ closeButton: false, closeOnClick: false }, true);
      this._eventHandlers = [];
    }

    eachLayer(callback) {
      this.features.forEach(callback);
    }

    getBounds() {
      return this.bounds;
    }

    setStyle(style) {
      this.features.forEach((layer) => {
        layer.options = typeof style === "function" ? (style(layer.feature) || {}) : { ...(style || {}) };
        if (this.map?.raw.getSource(this.sourceId)) {
          this.map.raw.setFeatureState(
            { source: this.sourceId, id: layer.featureId },
            {
              fillColor: layer.options.fillColor || "#e5e7eb",
              fillOpacity: Number(layer.options.fillOpacity) || 0,
              lineColor: layer.options.color || "#000000",
              lineOpacity: Number(layer.options.opacity) || 0,
              lineWidth: Number(layer.options.weight) || 0,
            },
          );
        }
      });
      this.map?._markRenderPending();
      return this;
    }

    applyFeatureStates() {
      this.setStyle(this.options.style || ((feature) => {
        const layer = this.features.find((candidate) => candidate.feature === feature);
        return layer?.options || {};
      }));
    }

    featureForEvent(event) {
      const id = String(event?.features?.[0]?.id ?? event?.features?.[0]?.properties?.__lucidum_feature_id ?? "");
      return this.features.find((layer) => layer.featureId === id) || null;
    }

    openFeaturePopup(layer, lngLat = null) {
      if (!layer || !this.map) return false;
      const center = lngLat ? latLng(lngLat) : layer.getBounds().getCenter();
      const popup = new PopupLayer(layer._popupOptions || {});
      layer._popup = popup;
      popup.setLatLng(center);
      popup.setContent(resolveContent(layer._popupContent, layer));
      popup.openOn(this.map);
      return true;
    }

    prepareFeatureProperties() {
      this.features.forEach((layer) => {
        const properties = layer.feature.properties;
        properties.__lucidum_fill_color = layer.options.fillColor || "#e5e7eb";
        properties.__lucidum_fill_opacity = Number(layer.options.fillOpacity) || 0;
        properties.__lucidum_line_color = layer.options.color || "#000000";
        properties.__lucidum_line_opacity = Number(layer.options.opacity) || 0;
        properties.__lucidum_line_width = Number(layer.options.weight) || 0;
      });
    }

    bindStyleEvents(map) {
      const raw = map.raw;
      this._eventHandlers.forEach(([type, handler]) => raw.off(type, this.fillLayerId, handler));
      this._eventHandlers = [];
      if (!raw.getLayer(this.fillLayerId)) return;
      const mouseMove = (event) => {
        const layer = this.featureForEvent(event);
        if (!layer?._tooltipContent) return;
        raw.getCanvas().style.cursor = "pointer";
        this.tooltip
          .setLatLng({ lat: event.lngLat.lat, lng: event.lngLat.lng })
          .setContent(resolveContent(layer._tooltipContent, layer))
          .addTo(map);
      };
      const mouseLeave = () => {
        raw.getCanvas().style.cursor = "";
        this.tooltip.remove();
      };
      const click = (event) => {
        const layer = this.featureForEvent(event);
        if (!layer?._popupContent) return;
        this.openFeaturePopup(layer, { lat: event.lngLat.lat, lng: event.lngLat.lng });
      };
      raw.on("mousemove", this.fillLayerId, mouseMove);
      raw.on("mouseleave", this.fillLayerId, mouseLeave);
      raw.on("click", this.fillLayerId, click);
      this._eventHandlers = [
        ["mousemove", mouseMove],
        ["mouseleave", mouseLeave],
        ["click", click],
      ];
    }

    restoreStyleResources(map = this.map) {
      if (!map || this.map !== map) return this;
      const raw = map.raw;
      if (raw.getSource(this.sourceId)) return this;
      this.prepareFeatureProperties();
      raw.addSource(this.sourceId, {
        type: "geojson",
        data: this.geoJson,
        promoteId: "__lucidum_feature_id",
      });
      raw.addLayer({
        id: this.fillLayerId,
        type: "fill",
        source: this.sourceId,
        paint: {
          "fill-antialias": this.options.fillAntialias !== false,
          "fill-color": [
            "coalesce",
            ["feature-state", "fillColor"],
            ["get", "__lucidum_fill_color"],
            "#e5e7eb",
          ],
          "fill-opacity": [
            "coalesce",
            ["feature-state", "fillOpacity"],
            ["get", "__lucidum_fill_opacity"],
            0,
          ],
        },
      });
      raw.addLayer({
        id: this.lineLayerId,
        type: "line",
        source: this.sourceId,
        paint: {
          "line-color": [
            "coalesce",
            ["feature-state", "lineColor"],
            ["get", "__lucidum_line_color"],
            "#000000",
          ],
          "line-opacity": [
            "coalesce",
            ["feature-state", "lineOpacity"],
            ["get", "__lucidum_line_opacity"],
            0,
          ],
          "line-width": [
            "coalesce",
            ["feature-state", "lineWidth"],
            ["get", "__lucidum_line_width"],
            0,
          ],
        },
      });
      this.bindStyleEvents(map);
      this.applyFeatureStates();
      map._markSourcePending(this.sourceId);
      return this;
    }

    onAdd(map) {
      this.features.forEach((layer) => map._registerFeatureLayer(layer));
      const add = () => this.restoreStyleResources(map);
      if (map.raw.isStyleLoaded?.()) add();
      else map.whenStyleReady().then(add).catch(() => {});
    }

    onRemove(map) {
      this.tooltip.remove();
      this.features.forEach((layer) => {
        layer._popup?.remove();
        map._unregisterLayer(layer);
      });
      this._eventHandlers.forEach(([type, handler]) => map.raw.off(type, this.fillLayerId, handler));
      this._eventHandlers = [];
      if (map.raw.getLayer(this.lineLayerId)) map.raw.removeLayer(this.lineLayerId);
      if (map.raw.getLayer(this.fillLayerId)) map.raw.removeLayer(this.fillLayerId);
      if (map.raw.getSource(this.sourceId)) map.raw.removeSource(this.sourceId);
    }
  }

  class LayerGroup extends LayerBase {
    initialize() {
      this.layers = [];
    }

    addLayer(layer) {
      if (!this.layers.includes(layer)) this.layers.push(layer);
      if (this.map) layer.addTo(this.map);
      return this;
    }

    onAdd(map) {
      this.layers.forEach((layer) => layer.addTo(map));
    }

    onRemove() {
      this.layers.forEach((layer) => layer.remove());
    }
  }

  class MarkerLayer extends LayerBase {
    initialize(position, options = {}) {
      this.position = latLng(position);
      this.options = { ...options };
      this.rawMarker = null;
    }

    onAdd(map) {
      const element = document.createElement("div");
      const icon = this.options.icon || {};
      element.className = icon.options?.className || icon.className || "";
      element.innerHTML = icon.options?.html || icon.html || "";
      if (this.options.interactive === false) element.style.pointerEvents = "none";
      this.rawMarker = new maplibregl.Marker({ element, anchor: "center" })
        .setLngLat(lngLatLike(this.position))
        .addTo(map.raw);
    }

    onRemove() {
      this.rawMarker?.remove();
      this.rawMarker = null;
    }
  }

  class DivIcon {
    constructor(options = {}) {
      this.options = { ...options };
    }
  }

  class ControlBase {
    constructor(options = {}) {
      this.options = { ...(this.options || {}), ...options };
      this.map = null;
      this.rawControl = null;
    }

    addTo(map) {
      this.map = map;
      this.rawControl = {
        onAdd: () => this.onAdd(map),
        onRemove: () => this.onRemove?.(map),
      };
      const position = String(this.options.position || "topright").replace("top", "top-").replace("bottom", "bottom-");
      map.raw.addControl(this.rawControl, position);
      return this;
    }

    static extend(definition = {}) {
      class ExtendedControl extends this {}
      Object.assign(ExtendedControl.prototype, definition);
      return ExtendedControl;
    }
  }

  const DomUtil = {
    create(tagName, className = "", parent = null) {
      const element = document.createElement(tagName);
      element.className = className;
      parent?.appendChild(element);
      return element;
    },
  };

  const DomEvent = {
    disableClickPropagation(element) {
      ["click", "dblclick", "mousedown", "touchstart", "pointerdown"].forEach((type) => {
        element.addEventListener(type, (event) => event.stopPropagation());
      });
      return this;
    },
    disableScrollPropagation(element) {
      element.addEventListener("wheel", (event) => event.stopPropagation());
      return this;
    },
  };

  return {
    version: maplibregl.getVersion?.() || "",
    map(container, options) {
      return new MapFacade(container, options);
    },
    tileLayer(url, options) {
      return new RasterLayer(url, options);
    },
    canvasLayer(canvas, coordinates, options) {
      return new CanvasLayer(canvas, coordinates, options);
    },
    geoJSON(data, options) {
      return new GeoJsonLayer(data, options);
    },
    latLng,
    latLngBounds(value) {
      return new Bounds(value);
    },
    popup(options) {
      return new PopupLayer(options);
    },
    tooltip(options) {
      return new PopupLayer(options, true);
    },
    layerGroup() {
      return new LayerGroup();
    },
    marker(position, options) {
      return new MarkerLayer(position, options);
    },
    divIcon(options) {
      return new DivIcon(options);
    },
    Layer: LayerBase,
    Control: ControlBase,
    DomUtil,
    DomEvent,
    maplibregl,
  };
}
