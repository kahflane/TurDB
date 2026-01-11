-- Subquery Stress Test Suite
-- Tests various combinations of subqueries, ORDER BY, LIMIT, filters, functions, and expressions
-- Each query has an expected row count comment

-- =============================================================================
-- SECTION 1: Basic Subquery Tests
-- =============================================================================

-- Q1: Simple subquery with LIMIT
SELECT * FROM (SELECT id, name FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q2: Subquery with ORDER BY DESC + LIMIT (critical test)
SELECT * FROM (SELECT id, name FROM organizations ORDER BY id DESC LIMIT 3);
-- EXPECT_ROWS: 3

-- Q3: Subquery with ORDER BY ASC + LIMIT
SELECT * FROM (SELECT id, name FROM organizations ORDER BY id ASC LIMIT 3);
-- EXPECT_ROWS: 3

-- Q4: Verify ORDER BY DESC returns highest ID first
SELECT id FROM (SELECT id FROM organizations ORDER BY id DESC LIMIT 1);
-- EXPECT_MAX_ID

-- Q5: Verify ORDER BY ASC returns lowest ID first
SELECT id FROM (SELECT id FROM organizations ORDER BY id ASC LIMIT 1);
-- EXPECT_MIN_ID

-- =============================================================================
-- SECTION 2: Nested Subqueries (Multiple Levels)
-- =============================================================================

-- Q6: 2-level nesting - inner DESC, outer takes subset
SELECT * FROM (SELECT * FROM (SELECT id, name FROM organizations ORDER BY id DESC LIMIT 20) AS inner_sub LIMIT 5) AS outer_sub;
-- EXPECT_ROWS: 5

-- Q7: 2-level nesting - inner ASC, outer reverses
SELECT * FROM (SELECT * FROM (SELECT id, name FROM organizations ORDER BY id ASC LIMIT 50) AS inner_sub ORDER BY id DESC LIMIT 10) AS outer_sub;
-- EXPECT_ROWS: 10

-- Q8: 3-level deep nesting
SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT id, name FROM organizations LIMIT 100) AS l1 LIMIT 50) AS l2 LIMIT 10) AS l3;
-- EXPECT_ROWS: 10

-- Q9: 3-level with alternating ORDER directions
SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT id, name FROM organizations ORDER BY id DESC LIMIT 100) AS l1 ORDER BY id ASC LIMIT 30) AS l2 ORDER BY id DESC LIMIT 5) AS l3;
-- EXPECT_ROWS: 5

-- =============================================================================
-- SECTION 3: Filters (WHERE clauses)
-- =============================================================================

-- Q10: Subquery with simple WHERE
SELECT * FROM (SELECT id, name FROM organizations WHERE id > 1000 LIMIT 10);
-- EXPECT_ROWS: 10

-- Q11: Subquery with WHERE and ORDER BY DESC
SELECT * FROM (SELECT id, name FROM organizations WHERE id > 500 ORDER BY id DESC LIMIT 5);
-- EXPECT_ROWS: 5

-- Q12: WHERE with BETWEEN
SELECT * FROM (SELECT id, name FROM organizations WHERE id BETWEEN 100 AND 500 ORDER BY id LIMIT 10);
-- EXPECT_ROWS: 10

-- Q13: WHERE with LIKE pattern (starts with)
SELECT * FROM (SELECT id, name FROM organizations WHERE name LIKE 'A%' LIMIT 10);
-- EXPECT_ROWS: >= 1

-- Q14: WHERE with LIKE pattern (contains)
SELECT * FROM (SELECT id, name FROM organizations WHERE name LIKE '%Inc%' LIMIT 10);
-- EXPECT_ROWS: >= 1

-- Q15: WHERE with IS NULL
SELECT * FROM (SELECT id, name, description FROM organizations WHERE description IS NULL LIMIT 5);
-- EXPECT_ROWS: >= 1

-- Q16: WHERE with IS NOT NULL
SELECT * FROM (SELECT id, name, description FROM organizations WHERE description IS NOT NULL LIMIT 5);
-- EXPECT_ROWS: >= 1

-- Q17: WHERE with multiple AND conditions
SELECT * FROM (SELECT id, name FROM organizations WHERE id > 100 AND id < 500 AND name LIKE 'A%' LIMIT 10);
-- EXPECT_ROWS: >= 0

-- Q18: Outer WHERE on subquery result
SELECT * FROM (SELECT id, name FROM organizations ORDER BY id DESC LIMIT 100) AS sub WHERE id > 3800;
-- EXPECT_ROWS: >= 1

-- Q19: Complex filter chain
SELECT * FROM (SELECT * FROM (SELECT id, name FROM organizations WHERE id > 10 ORDER BY id DESC LIMIT 500) AS inner_sub WHERE id < 3500 LIMIT 20) AS outer_sub;
-- EXPECT_ROWS: 20

-- =============================================================================
-- SECTION 4: Aggregate Functions
-- =============================================================================

-- Q20: COUNT(*) in subquery
SELECT * FROM (SELECT COUNT(*) as total FROM organizations);
-- EXPECT_ROWS: 1

-- Q21: MAX in subquery
SELECT * FROM (SELECT MAX(id) as max_id FROM organizations);
-- EXPECT_ROWS: 1

-- Q22: MIN in subquery
SELECT * FROM (SELECT MIN(id) as min_id FROM organizations);
-- EXPECT_ROWS: 1

-- Q23: Multiple aggregates in subquery
SELECT * FROM (SELECT COUNT(*) as cnt, MIN(id) as min_id, MAX(id) as max_id FROM organizations);
-- EXPECT_ROWS: 1

-- Q24: Aggregate on subquery result
SELECT COUNT(*), MIN(id), MAX(id) FROM (SELECT id FROM organizations ORDER BY id DESC LIMIT 100);
-- EXPECT_ROWS: 1

-- Q25: SUM of computed column
SELECT * FROM (SELECT SUM(id) as id_sum FROM organizations WHERE id < 100);
-- EXPECT_ROWS: 1

-- Q26: AVG in subquery
SELECT * FROM (SELECT AVG(id) as avg_id FROM organizations);
-- EXPECT_ROWS: 1

-- Q27: GROUP BY with aggregate
SELECT * FROM (SELECT creation_date, COUNT(*) as cnt FROM organizations GROUP BY creation_date LIMIT 10);
-- EXPECT_ROWS: 10

-- Q28: GROUP BY with ORDER BY on aggregate
SELECT * FROM (SELECT creation_date, COUNT(*) as cnt FROM organizations GROUP BY creation_date ORDER BY cnt DESC LIMIT 5);
-- EXPECT_ROWS: 5

-- =============================================================================
-- SECTION 5: Expressions and Computed Columns
-- =============================================================================

-- Q29: Arithmetic expression in SELECT
SELECT * FROM (SELECT id, id * 2 as doubled_id, id + 100 as plus_hundred FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q30: String concatenation
SELECT * FROM (SELECT id, name || ' - ' || slug as combined FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q31: CASE expression
SELECT * FROM (SELECT id, CASE WHEN id < 100 THEN 'low' WHEN id < 1000 THEN 'medium' ELSE 'high' END as category FROM organizations LIMIT 10);
-- EXPECT_ROWS: 10

-- Q32: COALESCE for NULL handling
SELECT * FROM (SELECT id, COALESCE(description, 'No description') as desc FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q33: Multiple expressions combined
SELECT * FROM (SELECT id, name, id * 10 as score, CASE WHEN id % 2 = 0 THEN 'even' ELSE 'odd' END as parity FROM organizations LIMIT 10);
-- EXPECT_ROWS: 10

-- =============================================================================
-- SECTION 6: Column Selection and Aliases
-- =============================================================================

-- Q34: Select specific columns from subquery
SELECT id FROM (SELECT id, name, slug FROM organizations ORDER BY id DESC LIMIT 10);
-- EXPECT_ROWS: 10

-- Q35: Column aliases in subquery
SELECT org_id, org_name FROM (SELECT id as org_id, name as org_name FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- Q36: Aliased subquery with qualified column access
SELECT sub.id, sub.name FROM (SELECT id, name FROM organizations ORDER BY id ASC LIMIT 5) AS sub;
-- EXPECT_ROWS: 5

-- Q37: SELECT * with column subset in inner
SELECT * FROM (SELECT id, name FROM organizations LIMIT 5);
-- EXPECT_ROWS: 5

-- =============================================================================
-- SECTION 7: ORDER BY Variations
-- =============================================================================

-- Q38: ORDER BY multiple columns
SELECT * FROM (SELECT id, name, creation_date FROM organizations ORDER BY creation_date DESC, id ASC LIMIT 5);
-- EXPECT_ROWS: 5

-- Q39: ORDER BY with expression
SELECT * FROM (SELECT id, name, id % 10 as mod_ten FROM organizations ORDER BY id % 10, id LIMIT 10);
-- EXPECT_ROWS: 10

-- Q40: ORDER BY on outer query
SELECT * FROM (SELECT id, name FROM organizations LIMIT 20) AS sub ORDER BY id DESC LIMIT 5;
-- EXPECT_ROWS: 5

-- Q41: ORDER BY text column
SELECT * FROM (SELECT id, name FROM organizations ORDER BY name LIMIT 10);
-- EXPECT_ROWS: 10

-- Q42: ORDER BY text DESC
SELECT * FROM (SELECT id, name FROM organizations ORDER BY name DESC LIMIT 10);
-- EXPECT_ROWS: 10

-- =============================================================================
-- SECTION 8: LIMIT and OFFSET
-- =============================================================================

-- Q43: OFFSET without ORDER BY
SELECT * FROM (SELECT id, name FROM organizations LIMIT 5 OFFSET 10);
-- EXPECT_ROWS: 5

-- Q44: ORDER BY + LIMIT + OFFSET
SELECT * FROM (SELECT id, name FROM organizations ORDER BY id ASC LIMIT 5 OFFSET 10);
-- EXPECT_ROWS: 5

-- Q45: Large OFFSET (may return fewer rows or zero)
SELECT * FROM (SELECT id FROM organizations LIMIT 10 OFFSET 3870);
-- EXPECT_ROWS: >= 0

-- Q46: OFFSET at exact boundary
SELECT * FROM (SELECT id FROM organizations ORDER BY id DESC LIMIT 5 OFFSET 0);
-- EXPECT_ROWS: 5

-- =============================================================================
-- SECTION 9: DISTINCT
-- =============================================================================

-- Q47: DISTINCT in subquery
SELECT * FROM (SELECT DISTINCT creation_date FROM organizations LIMIT 10);
-- EXPECT_ROWS: 10

-- Q48: DISTINCT with ORDER BY
SELECT * FROM (SELECT DISTINCT creation_date FROM organizations ORDER BY creation_date DESC LIMIT 5);
-- EXPECT_ROWS: 5

-- =============================================================================
-- SECTION 10: Edge Cases
-- =============================================================================

-- Q49: Empty result (impossible WHERE)
SELECT * FROM (SELECT id, name FROM organizations WHERE id < 0 LIMIT 5);
-- EXPECT_ROWS: 0

-- Q50: LIMIT 1 edge case
SELECT * FROM (SELECT id, name FROM organizations ORDER BY id DESC LIMIT 1);
-- EXPECT_ROWS: 1

-- Q51: Large LIMIT (more than table has)
SELECT COUNT(*) FROM (SELECT id FROM organizations LIMIT 100000);
-- EXPECT_ROWS: 1

-- Q52: Subquery returns all columns with star
SELECT * FROM (SELECT * FROM organizations ORDER BY id DESC LIMIT 5);
-- EXPECT_ROWS: 5

-- Q53: Window function ROW_NUMBER
SELECT * FROM (SELECT id, name, ROW_NUMBER() OVER (ORDER BY id) as rn FROM organizations LIMIT 10);
-- EXPECT_ROWS: 10

-- Q54: Nested with window function and LIMIT
SELECT * FROM (SELECT * FROM (SELECT id, ROW_NUMBER() OVER (ORDER BY id DESC) as rn FROM organizations LIMIT 20) AS inner_sub LIMIT 10) AS outer_sub;
-- EXPECT_ROWS: 10

-- =============================================================================
-- SECTION 11: Complex Combined Queries
-- =============================================================================

-- Q55: Filter + Aggregate + Subquery
SELECT * FROM (SELECT COUNT(*) as cnt, MAX(id) as max_id FROM (SELECT id FROM organizations WHERE id > 1000 LIMIT 500));
-- EXPECT_ROWS: 1

-- Q56: Multiple levels with different operations
SELECT final.id FROM (
    SELECT * FROM (
        SELECT id, name FROM organizations WHERE id > 100 ORDER BY id DESC LIMIT 200
    ) AS mid WHERE id < 3000 ORDER BY id ASC LIMIT 50
) AS final ORDER BY id DESC LIMIT 10;
-- EXPECT_ROWS: 10

-- Q57: Aggregate on filtered nested subquery
SELECT SUM(id) as total FROM (SELECT id FROM (SELECT id FROM organizations WHERE id < 100 ORDER BY id LIMIT 50) AS inner_sub);
-- EXPECT_ROWS: 1

-- Q58: Complex expression with nested subquery
SELECT * FROM (
    SELECT id,
           name,
           CASE WHEN id < 500 THEN 'small' ELSE 'large' END as size,
           id * 2 as doubled
    FROM organizations
    WHERE id > 50
    ORDER BY id DESC
    LIMIT 10
);
-- EXPECT_ROWS: 10

-- Q59: Deeply nested with aggregates at different levels
SELECT MAX(inner_cnt) as max_count FROM (
    SELECT creation_date, COUNT(*) as inner_cnt
    FROM organizations
    GROUP BY creation_date
    ORDER BY inner_cnt DESC
    LIMIT 20
);
-- EXPECT_ROWS: 1

-- Q60: Cross join of subqueries (cartesian product)
SELECT a.id, b.id FROM (SELECT id FROM organizations LIMIT 3) AS a, (SELECT id FROM organizations LIMIT 3) AS b;
-- EXPECT_ROWS: 9
