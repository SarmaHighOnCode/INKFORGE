/** @type {import('tailwindcss').Config} */
export default {
    content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
    theme: {
        extend: {
            colors: {
                ink: {
                    black: "#1a1a2e",
                    blue: "#0066cc",
                    "dark-blue": "#003366",
                    sepia: "#704214",
                },
                paper: {
                    white: "#fefefe",
                    cream: "#f5f0e8",
                    aged: "#e8dcc8",
                    lined: "#f0f0ff",
                },
            },
            fontFamily: {
                sans: ["Inter", "system-ui", "sans-serif"],
                display: ["Outfit", "system-ui", "sans-serif"],
            },
        },
    },
    plugins: [],
};
