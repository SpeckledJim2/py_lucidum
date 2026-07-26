let echartsGlPromise = null;

export function isEchartsTargetReady(target) {
  return Boolean(
    target?.isConnected
    && Number(target.clientWidth) > 0
    && Number(target.clientHeight) > 0
  );
}

export async function ensureEchartsGl(plotType = "surface") {
  if (plotType !== "surface") return false;
  if (window.__lucidumEchartsGlLoaded) return false;
  if (!echartsGlPromise) {
    echartsGlPromise = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "/static/vendor/echarts-gl/echarts-gl.min.js";
      script.onload = () => {
        window.__lucidumEchartsGlLoaded = true;
        resolve();
      };
      script.onerror = () => reject(new Error("ECharts GL did not load"));
      document.head.append(script);
    });
  }
  await echartsGlPromise;
  return true;
}
