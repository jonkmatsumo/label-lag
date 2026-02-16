import { test, expect } from '@playwright/test';

test.describe('Dataset Profiles Page', () => {
    test.beforeEach(async ({ page }) => {
        await page.route('**/bff/v1/dataset/profiles*', async (route) => {
            await route.fulfill({
                json: {
                    profiles: [
                        {
                            profile_id: 'prof-1',
                            created_at: new Date().toISOString(),
                            row_count: 1000,
                            column_count: 20,
                            size_bytes: 1024 * 1024,
                        },
                        {
                            profile_id: 'prof-2',
                            created_at: new Date().toISOString(),
                            row_count: 1100,
                            column_count: 20,
                            size_bytes: 1024 * 1024 * 1.1,
                        }
                    ],
                    pagination: {
                        total: 2,
                        next_cursor: null,
                    },
                },
            });
        });

        await page.goto('/dataset/profiles');
    });

    test('displays profiles list', async ({ page }) => {
        await expect(page.getByRole('heading', { name: 'Dataset Profiles', level: 1 })).toBeVisible();
        await expect(page.getByText('prof-1')).toBeVisible();
        await expect(page.getByText('prof-2')).toBeVisible();
    });

    test('compares selected profiles', async ({ page }) => {
        // Mock compare response
        await page.route('**/bff/v1/dataset/profiles/compare*', async (route) => {
            await route.fulfill({
                json: {
                    features: [
                        { feature: 'amount', psi: 0.15, severity: 'medium' },
                        { feature: 'age', psi: 0.01, severity: 'low' },
                    ]
                }
            });
        });

        // Select two profiles
        const checkboxes = page.locator('input[type="checkbox"]');
        await checkboxes.first().waitFor();
        const count = await checkboxes.count();
        expect(count).toBeGreaterThanOrEqual(2);

        await checkboxes.nth(0).check();
        await checkboxes.nth(1).check();

        // Click compare
        await page.getByRole('button', { name: 'Compare (2/2)' }).click();

        // Verify navigation and content
        await expect(page).toHaveURL(/\/dataset\/profiles\/compare/);
        await expect(page.getByRole('heading', { name: 'Compare Profiles', level: 1 })).toBeVisible();
        await expect(page.getByText('amount')).toBeVisible();
        await expect(page.getByText('MEDIUM')).toBeVisible();
    });
});
