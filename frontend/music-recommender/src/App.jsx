export default function MusicRecommenderPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-zinc-900 to-zinc-800 text-white p-8">
      <div className="max-w-7xl mx-auto">

        {/* Header */}
        <div className="mb-10 text-center">
          <h1 className="text-5xl font-bold mb-4 tracking-tight">
            Music Recommendation System
          </h1>

          <p className="text-zinc-300 text-lg max-w-3xl mx-auto leading-relaxed">
            Explore musical similarity using machine learning models based on
            acoustic characteristics, genre patterns, rhythm structures and
            audio features.
          </p>
        </div>

        {/* Filtros */}
        <div className="bg-zinc-900/70 backdrop-blur rounded-3xl p-6 shadow-2xl border border-zinc-700 mb-8">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">

            <input
              type="text"
              placeholder="Search artist or band"
              className="bg-zinc-800 border border-zinc-700 rounded-2xl p-3 outline-none focus:ring-2 focus:ring-white text-white"
            />

            <select className="bg-zinc-800 border border-zinc-700 rounded-2xl p-3 text-white">
              <option>Musical Style</option>
              <option>Rock</option>
              <option>Metal</option>
              <option>Pop</option>
              <option>Jazz</option>
              <option>Hip Hop</option>
            </select>

            <select className="bg-zinc-800 border border-zinc-700 rounded-2xl p-3 text-white">
              <option>Music Type</option>
              <option>Energetic</option>
              <option>Melancholic</option>
              <option>Acoustic</option>
              <option>Electronic</option>
              <option>Experimental</option>
            </select>

            <select className="bg-zinc-800 border border-zinc-700 rounded-2xl p-3 text-white">
              <option>Popularity</option>
              <option>Mainstream</option>
              <option>Underground</option>
              <option>Trending</option>
            </select>

            <button className="bg-white text-black font-semibold rounded-2xl p-3 hover:scale-105 transition-all duration-300 shadow-lg">
              Generate Recommendations
            </button>
          </div>
        </div>

        {/* Dashboard */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">

          <div className="bg-zinc-900/70 border border-zinc-700 rounded-3xl p-6 shadow-xl">
            <h2 className="text-2xl font-bold mb-4">
              Model Insights
            </h2>

            <div className="space-y-4 text-zinc-300">
              <div>
                <p className="text-sm uppercase tracking-wider text-zinc-500">
                  Active Model
                </p>
                <p className="text-xl font-semibold text-white">
                  Experimental Similarity Model
                </p>
              </div>

              <div>
                <p className="text-sm uppercase tracking-wider text-zinc-500">
                  Features Analyzed
                </p>
                <p>
                  Tempo, MFCCs, Spectral Features, RMS Energy,
                  Chroma, Rhythm Patterns
                </p>
              </div>

              <div>
                <p className="text-sm uppercase tracking-wider text-zinc-500">
                  Recommendation Logic
                </p>
                <p>
                  Hybrid similarity approach using dimensionality reduction
                  and nearest-neighbor proximity.
                </p>
              </div>
            </div>
          </div>

          {/* Resultados */}
          <div className="lg:col-span-2 bg-zinc-900/70 border border-zinc-700 rounded-3xl p-6 shadow-xl">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold">
                Recommended Artists
              </h2>

              <span className="bg-zinc-800 px-4 py-2 rounded-full text-sm text-zinc-300">
                Top Similarities
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">

              <div className="bg-gradient-to-br from-zinc-800 to-zinc-900 border border-zinc-700 rounded-3xl p-5 hover:scale-[1.02] transition-all duration-300 shadow-lg">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-2xl font-bold">Arctic Monkeys</h3>
                  <span className="text-green-400 font-semibold">95%</span>
                </div>

                <p className="text-zinc-400 mb-4">
                  Strong similarity in guitar texture, rhythm energy and vocal style.
                </p>

                <div className="flex flex-wrap gap-2">
                  <span className="bg-zinc-700 px-3 py-1 rounded-full text-sm">
                    Indie Rock
                  </span>

                  <span className="bg-zinc-700 px-3 py-1 rounded-full text-sm">
                    Alternative
                  </span>
                </div>
              </div>

              <div className="bg-gradient-to-br from-zinc-800 to-zinc-900 border border-zinc-700 rounded-3xl p-5 hover:scale-[1.02] transition-all duration-300 shadow-lg">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-2xl font-bold">The Strokes</h3>
                  <span className="text-green-400 font-semibold">92%</span>
                </div>

                <p className="text-zinc-400 mb-4">
                  Similar instrumental patterns and spectral characteristics.
                </p>

                <div className="flex flex-wrap gap-2">
                  <span className="bg-zinc-700 px-3 py-1 rounded-full text-sm">
                    Garage Rock
                  </span>

                  <span className="bg-zinc-700 px-3 py-1 rounded-full text-sm">
                    Indie
                  </span>
                </div>
              </div>

              <div className="bg-gradient-to-br from-zinc-800 to-zinc-900 border border-zinc-700 rounded-3xl p-5 hover:scale-[1.02] transition-all duration-300 shadow-lg">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-2xl font-bold">Radiohead</h3>
                  <span className="text-green-400 font-semibold">89%</span>
                </div>

                <p className="text-zinc-400 mb-4">
                  Similar harmonic density and atmospheric composition.
                </p>

                <div className="flex flex-wrap gap-2">
                  <span className="bg-zinc-700 px-3 py-1 rounded-full text-sm">
                    Experimental
                  </span>

                  <span className="bg-zinc-700 px-3 py-1 rounded-full text-sm">
                    Alternative Rock
                  </span>
                </div>
              </div>

              <div className="bg-gradient-to-br from-zinc-800 to-zinc-900 border border-zinc-700 rounded-3xl p-5 hover:scale-[1.02] transition-all duration-300 shadow-lg">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-2xl font-bold">Muse</h3>
                  <span className="text-green-400 font-semibold">87%</span>
                </div>

                <p className="text-zinc-400 mb-4">
                  High rhythmic similarity and energetic frequency signatures.
                </p>

                <div className="flex flex-wrap gap-2">
                  <span className="bg-zinc-700 px-3 py-1 rounded-full text-sm">
                    Progressive Rock
                  </span>

                  <span className="bg-zinc-700 px-3 py-1 rounded-full text-sm">
                    Alternative
                  </span>
                </div>
              </div>

            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-zinc-500 text-sm mt-10">
          Machine Learning Music Recommendation Platform • Academic Project
        </div>
      </div>
    </div>
  )
}
