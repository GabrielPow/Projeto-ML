export default function MusicRecommenderPage() {
  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-6xl mx-auto">

        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold mb-2">
            Sistema de Recomendação Musical
          </h1>

          <p className="text-gray-600 text-lg">
            PCA + KNN + Cosine Similarity
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* Painel lateral */}
          <div className="bg-white rounded-2xl shadow-lg p-6">

            <h2 className="text-2xl font-semibold mb-4">
              Buscar Banda
            </h2>

            <input
              type="text"
              placeholder="Digite uma banda..."
              className="w-full border rounded-xl p-3 mb-4"
            />

            <button
              className="w-full bg-black text-white rounded-xl p-3 hover:opacity-90 transition"
            >
              Gerar Recomendação
            </button>

            <div className="mt-8">
              <h3 className="font-semibold text-lg mb-3">
                Como funciona?
              </h3>

              <p className="text-gray-600 text-sm leading-relaxed">
                O sistema utiliza PCA para redução dimensional e KNN com
                cosine similarity para encontrar artistas musicalmente
                semelhantes com base em características acústicas.
              </p>
            </div>
          </div>

          {/* Área principal */}
          <div className="lg:col-span-2 bg-white rounded-2xl shadow-lg p-6">

            <h2 className="text-2xl font-semibold mb-6">
              Recomendações
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

              {/* Card exemplo */}
              <div className="border rounded-2xl p-5 hover:shadow-md transition">
                <h3 className="text-xl font-bold mb-2">
                  Metallica
                </h3>

                <p className="text-gray-600 mb-3">
                  Similaridade: 94%
                </p>

                <div className="flex flex-wrap gap-2">
                  <span className="bg-gray-100 px-3 py-1 rounded-full text-sm">
                    Metal
                  </span>

                  <span className="bg-gray-100 px-3 py-1 rounded-full text-sm">
                    Heavy Metal
                  </span>
                </div>
              </div>

              <div className="border rounded-2xl p-5 hover:shadow-md transition">
                <h3 className="text-xl font-bold mb-2">
                  Megadeth
                </h3>

                <p className="text-gray-600 mb-3">
                  Similaridade: 91%
                </p>

                <div className="flex flex-wrap gap-2">
                  <span className="bg-gray-100 px-3 py-1 rounded-full text-sm">
                    Thrash Metal
                  </span>

                  <span className="bg-gray-100 px-3 py-1 rounded-full text-sm">
                    Metal
                  </span>
                </div>
              </div>

              <div className="border rounded-2xl p-5 hover:shadow-md transition">
                <h3 className="text-xl font-bold mb-2">
                  Slayer
                </h3>

                <p className="text-gray-600 mb-3">
                  Similaridade: 89%
                </p>

                <div className="flex flex-wrap gap-2">
                  <span className="bg-gray-100 px-3 py-1 rounded-full text-sm">
                    Thrash Metal
                  </span>
                </div>
              </div>

              <div className="border rounded-2xl p-5 hover:shadow-md transition">
                <h3 className="text-xl font-bold mb-2">
                  Pantera
                </h3>

                <p className="text-gray-600 mb-3">
                  Similaridade: 87%
                </p>

                <div className="flex flex-wrap gap-2">
                  <span className="bg-gray-100 px-3 py-1 rounded-full text-sm">
                    Groove Metal
                  </span>
                </div>
              </div>

            </div>
          </div>

        </div>

        {/* Rodapé */}
        <div className="mt-10 text-center text-gray-500 text-sm">
          Projeto acadêmico de recomendação musical utilizando Machine Learning.
        </div>

      </div>
    </div>
  )
}
