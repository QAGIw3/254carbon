import { test, expect } from '@playwright/test';

test.describe('User Journey Tests', () => {
  test('complete user journey: login to report generation', async ({ page }) => {
    // Navigate to application
    await page.goto('http://localhost:3000');

    // Should redirect to Keycloak login
    await expect(page).toHaveURL(/keycloak.*login/);

    // Mock login for testing (in real scenario, would use actual Keycloak)
    await page.evaluate(() => {
      // Simulate successful login by setting auth state
      localStorage.setItem('auth-token', 'test-jwt-token');
      localStorage.setItem('auth-user', JSON.stringify({
        sub: 'test-user',
        name: 'Test User',
        email: 'test@example.com'
      }));
    });

    // Navigate back to app after login
    await page.goto('http://localhost:3000');

    // Should be on dashboard
    await expect(page.locator('h1')).toContainText(/dashboard/i);

    // Navigate to Market Explorer
    await page.click('text=Market Explorer');

    // Should load market data
    await expect(page.locator('.market-heatmap')).toBeVisible();

    // Test heatmap interaction
    const heatmapCell = page.locator('.market-heatmap rect').first();
    await heatmapCell.click();

    // Should show drill-down info
    await expect(page.locator('text=Selected:')).toBeVisible();

    // Navigate to Curve Viewer
    await page.click('text=Curve Viewer');

    // Should load 3D visualization
    await expect(page.locator('canvas')).toBeVisible();

    // Test 3D controls
    await page.click('text=Wireframe');
    await page.click('text=Auto Rotate');

    // Navigate to Downloads/Reports
    await page.click('text=Downloads');

    // Generate a report
    await page.click('text=Generate Report');

    // Should show report generation form
    await expect(page.locator('input[type="date"]')).toBeVisible();

    // Fill out form and submit
    await page.fill('input[name="market"]', 'power');
    await page.fill('input[name="as_of_date"]', '2024-01-01');
    await page.click('button[type="submit"]');

    // Should show loading state then success
    await expect(page.locator('text=Generating report')).toBeVisible();

    // Wait for completion and check for download link
    await expect(page.locator('text=Download Report')).toBeVisible();
  });

  test('websocket real-time updates', async ({ page }) => {
    // Setup authenticated session
    await page.goto('http://localhost:3000');
    await page.evaluate(() => {
      localStorage.setItem('auth-token', 'test-jwt-token');
    });

    // Navigate to dashboard with real-time data
    await page.goto('http://localhost:3000/dashboard');

    // Should establish WebSocket connection
    await expect(page.locator('text=Connected')).toBeVisible();

    // Wait for price updates (mocked in test environment)
    await page.waitForTimeout(2000);

    // Should show live price data
    await expect(page.locator('text=MISO')).toBeVisible();
    await expect(page.locator('text=PJM')).toBeVisible();
  });

  test('responsive design and mobile interactions', async ({ page }) => {
    // Test mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });

    await page.goto('http://localhost:3000');
    await page.evaluate(() => {
      localStorage.setItem('auth-token', 'test-jwt-token');
    });

    // Should load mobile-optimized layout
    await expect(page.locator('nav')).toBeVisible();

    // Test mobile navigation
    await page.click('[data-testid="mobile-menu"]');

    // Should show mobile menu
    await expect(page.locator('[data-testid="mobile-nav"]')).toBeVisible();

    // Test touch interactions on heatmap
    await page.goto('http://localhost:3000/market-explorer');

    // Touch/click on heatmap cell
    const heatmapCell = page.locator('.market-heatmap rect').first();
    await heatmapCell.tap(); // Use tap for mobile

    // Should handle touch interaction
    await expect(page.locator('text=Selected:')).toBeVisible();
  });
});
