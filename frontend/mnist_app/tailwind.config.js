/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                burn: {
                    DEFAULT: '#e25822', // Flame color
                    dark: '#b23b1e',
                },
                pytorch: {
                    DEFAULT: '#ee4c2c', // PyTorch brand color approx
                    dark: '#c43016',
                }
            },
            fontFamily: {
                sans: ['Inter', 'sans-serif'],
            }
        },
    },
    plugins: [],
}
