/**
 * INKFORGE — Main Application Component
 *
 * Root component that assembles the handwriting synthesis UI:
 * - Text Input Panel
 * - Style & Parameter Controls
 * - Canvas Preview (animated stroke playback)
 * - Export Options
 */

function App() {
    return (
        <div className="min-h-screen bg-ink-black text-white">
            {/* TODO: Implement UI components */}
            {/* <Header /> */}
            {/* <TextInputPanel /> */}
            {/* <StyleSelector /> */}
            {/* <ParameterControls /> */}
            {/* <CanvasPreview /> */}
            {/* <ExportPanel /> */}

            <main className="flex items-center justify-center min-h-screen">
                <div className="text-center">
                    <h1 className="text-5xl font-display font-bold mb-4">
                        ✍ Inkforge
                    </h1>
                    <p className="text-xl text-gray-400">
                        Human-Like Handwriting Synthesis Engine
                    </p>
                    <p className="text-sm text-gray-600 mt-2">
                        Coming soon — LSTM+MDN stroke-level generation
                    </p>
                </div>
            </main>
        </div>
    );
}

export default App;
