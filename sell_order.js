const { ethers } = require('ethers');
const dotenv = require('dotenv');
const logger = require('./logger');
dotenv.config();

const provider = new ethers.JsonRpcProvider(process.env.INFURA_PROJECT_ID);
const wallet = new ethers.Wallet(process.env.PRIVATE_KEY, provider);

// Add ERC20 ABI definition
const ERC20_ABI = [
    "function balanceOf(address owner) view returns (uint256)",
    "function decimals() view returns (uint8)",
    "function symbol() view returns (string)",
    "function transfer(address to, uint256 amount) returns (bool)",
    "function approve(address spender, uint256 amount) returns (bool)"
];

const UNISWAP_ROUTER_ADDRESS = "0x4752ba5dbc23f44d87826276bf6fd6b1c372ad24"; // Uniswap V2 Router on Base
const UNISWAP_ROUTER_ABI = [
  "function swapExactTokensForTokens(uint amountIn, uint amountOutMin, address[] calldata path, address to, uint deadline) external returns (uint[] memory amounts)",
  "function getAmountsOut(uint amountIn, address[] calldata path) external view returns (uint[] memory amounts)",
];

const router = new ethers.Contract(
  UNISWAP_ROUTER_ADDRESS,
  UNISWAP_ROUTER_ABI,
  wallet
);

const WETH_ADDRESS = "0x4200000000000000000000000000000000000006"; // WETH on Base
const USDT_ADDRESS = "0xfde4C96c8593536E31F229EA8f37b2ADa2699bb2"; // USDT on Base

// Create USDT contract instance
const USDTContract = new ethers.Contract(USDT_ADDRESS, ERC20_ABI, wallet);

// Define the amount of USDT you're swapping
const amountInUSDT = ethers.parseEther("5.68"); // 0.001 ETH worth of USDT (0.001/0.000176)

async function swapExactTokensForWETH() {
  try {
    // Get wallet balances
    const initialWETHBalance = await provider.getBalance(wallet.address);
    const initialUSDTBalance = await USDTContract.balanceOf(wallet.address);

    console.log(`buy_order.js: Starting Wallet WETH Balance: ${ethers.formatEther(initialWETHBalance)} WETH`);
    console.log(`buy_order.js: Starting Wallet USDT Balance: ${ethers.formatUnits(initialUSDTBalance, 6)} USDT`);

    // Define the amount of USDT you're swapping
    const amountInUSDT = ethers.parseEther("5.68"); // 0.001 ETH worth of USDT

    // For approvals and transactions, use the contract with wallet
    await (await USDTContract.approve(UNISWAP_ROUTER_ADDRESS, amountInUSDT)).wait();

    // Get expected amount out
    const [, expectedAmountOut] = await router.getAmountsOut(amountInUSDT, [
      USDT_ADDRESS,
      WETH_ADDRESS,
    ]);
    const amountOutMin = (expectedAmountOut * BigInt(95)) / BigInt(100); // 5% slippage tolerance

    // Set a reasonable deadline (current time + 20 minutes)
    const deadline = Math.floor(Date.now() / 1000) + 60 * 20;

    // Define the path for the swap
    const path = [USDT_ADDRESS, WETH_ADDRESS];

    // Send the swap transaction
    const tx = await router.swapExactTokensForTokens(
      amountInUSDT,
      amountOutMin,
      path,
      wallet.address,
      deadline,
      { gasLimit: 300000 }
    );

    // Add logging
    logger.logTrade(
      "BUY",
      ethers.formatEther(amountInUSDT),
      "WETH",
      expectedAmountOut
    );

    console.log(`Transaction submitted: ${tx.hash}`);
    const receipt = await tx.wait();
    logger.logInfo(`Transaction mined in block ${receipt.blockNumber}`);
    // Get final balances
    const finalWETHBalance = await provider.getBalance(WETH_ADDRESS);
    const finalUSDTBalance = await provider.getBalance(USDT_ADDRESS);

    console.log(
      `buy_order.js: Final WETH Balance: ${ethers.formatEther(finalWETHBalance)} WETH`
    );
    console.log(
      `buy_order.js: Final USDT Balance: ${ethers.formatUnits(finalUSDTBalance, 6)} USDT`
    );

    // Calculate slippage
    const actualAmountOut = receipt.events.find((e) => e.event === "Swap")?.args
      ?.amount1;
    const slippage =
      (Number(expectedAmountOut - actualAmountOut) /
        Number(expectedAmountOut)) *
      100;
    logger.logInfo(`Slippage: ${slippage.toFixed(2)}%`);

    // If we get here, the swap was successful
    return { success: true, receipt: receipt };

  } catch (error) {
    logger.logError(error);
    console.error(`Error occurred during swap: ${error.message}`);
    return { success: false, reason: error.message };
  }
}

module.exports = { swapExactTokensForWETH };


  
