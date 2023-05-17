const mathjs = require("mathjs");

// Daily Price data
const prices = require("./price_data.json");

function calculateVaR(prices, riskLevel, collateralAmount, varPeriod) {
  // Calculate daily log returns
  let logReturns = [];
  for (let i = 1; i < prices.length; i++) {
    logReturns.push(Math.log(prices[i].rate_close / prices[i - 1].rate_close));
  }

  // Calculate daily volatility (standard deviation)
  const mean = logReturns.reduce((a, b) => a + b) / logReturns.length;
  const variance =
    logReturns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) /
    logReturns.length;
  const volatility = Math.sqrt(variance);

  // Calculate z-score corresponding to risk level
  const zScore = -1 * mathjs.quantileSeq([riskLevel], 0.5, true);

  // Calculate VaR
  const dailyVaR = collateralAmount * zScore * volatility;

  if (varPeriod === 0) {
    return dailyVaR;
  } else if (varPeriod === 1) {
    return dailyVaR * Math.sqrt(12); // Return monthly VaR
  } else if (varPeriod === 2) {
    return dailyVaR * Math.sqrt(365); // Return yearly VaR
  }
}

console.log(calculateVaR(prices, 0.01, 10000, 2));
