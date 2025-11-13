/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'meralco-orange': '#FF6F00',
        'meralco-black': '#000000',
        'meralco-gray': '#666666',
        'meralco-light-gray': '#E5E5E5',
        'meralco-white': '#FFFFFF',
        'meralco-dark-gray': '#333333',
      },
    },
  },
  plugins: [],
}

