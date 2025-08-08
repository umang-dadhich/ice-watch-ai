import { Link } from "react-router-dom";

const Index = () => {
  return (
    <div className="min-h-screen">
      <header className="container mx-auto pt-16 pb-8">
        <nav className="flex items-center justify-between">
          <a href="/" aria-label="GlacierEye Home" className="font-semibold tracking-tight text-lg">GlacierEye</a>
          <div className="flex items-center gap-3">
            <a className="btn-ghost px-4 py-2" href="#docs">Docs</a>
            <a className="btn-ghost px-4 py-2" href="https://streamlit.io" target="_blank" rel="noreferrer">Live Demo</a>
          </div>
        </nav>
      </header>

      <main className="container mx-auto px-4 pb-24">
        <section className="glacier-hero text-center px-6 py-16">
          <h1 className="text-4xl md:text-6xl font-bold tracking-tight mb-4 text-white">
            GlacierEye — Intelligent Glacier Segmentation
          </h1>
          <p className="max-w-3xl mx-auto text-white/90 text-lg md:text-xl mb-8">
            Research-grade, multi-spectral glacier mapping with temporal change monitoring, DEM-aware modeling, and explainability — built to win GlacierHack 2025.
          </p>
          <div className="flex items-center justify-center gap-4">
            <a className="btn-hero px-6 py-3 text-base font-medium" href="#get-started">Get Started</a>
            <a className="btn-ghost px-6 py-3 text-base font-medium" href="https://github.com" target="_blank" rel="noreferrer">GitHub</a>
          </div>
        </section>

        <section id="get-started" className="grid md:grid-cols-3 gap-6 mt-12">
          <article className="glacier-card p-6">
            <h2 className="text-xl font-semibold mb-2">Scientific Pipeline</h2>
            <p className="text-muted-foreground">Data ingestion (S2/L8/DEM), preprocessing, augmentation, training (U-Net/DeepLab/ViT), eval, and explainability.</p>
          </article>
          <article className="glacier-card p-6">
            <h2 className="text-xl font-semibold mb-2">Change Monitoring</h2>
            <p className="text-muted-foreground">Temporal stacks, area change in km², retreat heatmaps, baselines vs deep models.</p>
          </article>
          <article className="glacier-card p-6">
            <h2 className="text-xl font-semibold mb-2">Streamlit Demo</h2>
            <p className="text-muted-foreground">Upload scenes, toggle band composites/DEM, view masks, confidence, Grad-CAM, and download GeoTIFF.</p>
          </article>
        </section>

        <section id="docs" className="mt-12 text-center">
          <p className="text-sm text-muted-foreground">Full project code: Python modules, notebooks, and Streamlit app are included in this repo under glaciereye/ and streamlit_app/.</p>
        </section>
      </main>

      <footer className="container mx-auto pb-12 text-center text-sm text-muted-foreground">
        <p>© {new Date().getFullYear()} GlacierEye • Built for GlacierHack 2025</p>
      </footer>
    </div>
  );
};

export default Index;
